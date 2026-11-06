"""Validation, capability probing and native Ultralytics export helpers.

The GUI and compatibility probe import this module without loading CUDA. Actual
exports execute in run_model_export.py, in the selected managed Python process.
"""
from __future__ import annotations

from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import subprocess

EXPORT_FORMATS = ("onnx", "engine", "litert")
EXPORT_EXTENSIONS = {"onnx": ".onnx", "engine": ".engine", "litert": ".tflite"}
_WINDOWS_RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}


def normalize_export_model_name(value, fmt):
    """Return a filename stem or empty string for Ultralytics' default name.

    Users may enter a plain stem or include the suffix for the selected format.
    Validate against Windows filenames even on Linux to support shared/WSL paths.
    """
    if fmt not in EXPORT_EXTENSIONS:
        raise ValueError("Unknown export format: " + str(fmt))
    name = str(value or "").strip()
    if not name:
        return ""
    suffix = Path(name).suffix.lower()
    expected = EXPORT_EXTENSIONS[fmt]
    if suffix in EXPORT_EXTENSIONS.values():
        if suffix != expected:
            raise ValueError(f"The selected format uses {expected}, not {suffix}. Change the filename or export format.")
        name = name[:-len(suffix)]
    if (not name or name in (".", "..") or name.startswith(".")
            or name.endswith((" ", ".")) or any(ord(ch) < 32 for ch in name)
            or any(ch in '<>:"/\\|?*' for ch in name)):
        raise ValueError("Model name must be a filename, without folder paths or characters < > : \" / \\ | ? *.")
    if name.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("Model name is a reserved Windows filename; choose another name.")
    if len((name + expected).encode("utf-8")) > 240:
        raise ValueError("Model filename is too long (maximum 240 UTF-8 bytes including extension).")
    return name


def requested_export_filename(model_name, fmt):
    """Full requested file name, or empty string to keep exporter's default."""
    stem = normalize_export_model_name(model_name, fmt)
    return stem + EXPORT_EXTENSIONS[fmt] if stem else ""


def host_tensorrt_version(platform_name=None):
    """System libnvinfer package version, for diagnostic comparison only."""
    if (platform_name or sys.platform) != 'linux':
        return None
    for package in ('libnvinfer10', 'libnvinfer11', 'libnvinfer8'):
        try:
            result = subprocess.run(['dpkg-query', '-W', '-f=${Version}', package],
                                    capture_output=True, text=True, timeout=3, check=False)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('-', 1)[0]
        except (OSError, subprocess.TimeoutExpired):
            break
    return None


def is_jetson_environment(platform_name, machine, gpu=None):
    if platform_name != 'linux' or machine not in {'aarch64', 'arm64'}:
        return False
    markers = ('/etc/nv_tegra_release', '/etc/nvidia-container-runtime/host-files-for-container.d')
    return (any(Path(marker).exists() for marker in markers)
            or any(x in (gpu or '').lower() for x in ('thor', 'jetson', 'orin', 'xavier')))


def tensorrt_versions_match(python_version, host_version):
    if not python_version or not host_version:
        return None
    def numbers(text):
        return tuple(int(v) for v in re.findall(r'\d+', str(text))[:4])
    a, b = numbers(python_version), numbers(host_version)
    return a == b if a and b else None



def detect_export_formats():
    """Return format names actually advertised by this Ultralytics install."""
    from ultralytics.engine.exporter import export_formats
    table = export_formats()
    if hasattr(table, "to_dict"):
        try:
            formats = table["Argument"].tolist()
        except (KeyError, TypeError):
            formats = []
    elif isinstance(table, dict):
        formats = table.get("Argument", [])
    else:
        formats = []
    return {str(value).strip().lower() for value in formats}


def probe_export_environment(platform_name=None, machine=None):
    """Probe availability inside the *selected* managed Python interpreter."""
    platform_name = platform_name or sys.platform
    machine = (machine or platform.machine()).lower()
    report = {"python": sys.executable, "platform": platform_name, "machine": machine,
              "ultralytics": None, "formats": {}, "tensorrt": None,
              "cuda": False, "cuda_build": None, "gpu": None,
              "errors": {}, "install_package": "tensorrt",
              "host_tensorrt": host_tensorrt_version(platform_name),
              "tensorrt_version_match": None, "jetson": False}
    for name in EXPORT_FORMATS:
        report["formats"][name] = {"available": False, "reason": "Not checked"}
    try:
        import ultralytics
        report["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
        available = detect_export_formats()
    except Exception as exc:
        available = set()
        report["errors"]["ultralytics"] = f"{type(exc).__name__}: {exc}"

    if "onnx" in available:
        report["formats"]["onnx"] = {"available": True, "argument": "onnx", "reason": "Available"}
    else:
        report["formats"]["onnx"]["reason"] = "ONNX export is not advertised by this Ultralytics version."

    if platform_name == "linux" and machine in {"x86_64", "amd64"}:
        alias = "litert" if "litert" in available else ("tflite" if "tflite" in available else None)
        if alias:
            report["formats"]["litert"] = {
                "available": True, "argument": alias,
                "reason": "Ready" if alias == "litert" else "Legacy TFLite exporter (same .tflite format)",
            }
        else:
            report["formats"]["litert"]["reason"] = (
                "The installed Ultralytics version does not advertise LiteRT/TFLite export."
            )
    elif platform_name == "win32":
        # The Windows Conda install need not contain TensorFlow. LiteRT runs
        # in the user's separate Python 3.11 environment inside WSL2.
        if sys.platform == "win32":
            from wsl_litert import probe_wsl_litert
            wsl = probe_wsl_litert()
        else:
            wsl = {"available": False, "phase": "missing_wsl",
                   "reason": "WSL2 is needed on Windows for LiteRT export."}
        report["wsl"] = wsl
        if wsl.get("available"):
            report["formats"]["litert"] = {
                "available": True, "argument": wsl.get("argument", "litert"),
                "backend": "wsl", "reason": wsl["reason"],
            }
        else:
            report["formats"]["litert"]["reason"] = wsl["reason"]
    else:
        report["formats"]["litert"]["reason"] = (
            "LiteRT native export requires Linux x86-64. On Windows, install WSL2."
        )

    try:
        import torch
        report["cuda_build"] = str(torch.version.cuda or "")
        report["cuda"] = bool(torch.cuda.is_available())
        if report["cuda"]:
            report["gpu"] = torch.cuda.get_device_name(0)
        if report["cuda_build"].startswith("12"):
            report["install_package"] = "tensorrt-cu12"
        elif report["cuda_build"].startswith("13"):
            report["install_package"] = "tensorrt-cu13"
    except Exception as exc:
        report["errors"]["torch"] = f"{type(exc).__name__}: {exc}"
    report['jetson'] = is_jetson_environment(platform_name, machine, report['gpu'])
    if report['jetson']:
        # JetPack's package-managed TensorRT must never be replaced by a
        # PyPI wheel just to satisfy an export probe.
        report['install_package'] = None
    if not report["cuda"]:
        report["formats"]["engine"]["reason"] = "TensorRT export needs a CUDA-capable NVIDIA GPU in this environment."
    elif "engine" not in available:
        report["formats"]["engine"]["reason"] = "The installed Ultralytics version does not support TensorRT export."
    else:
        try:
            import tensorrt as trt
            if not callable(getattr(trt, "Builder", None)):
                raise RuntimeError("The TensorRT full builder bindings are unavailable (lean runtime is insufficient).")
            report["tensorrt"] = str(getattr(trt, "__version__", "unknown"))
            report['tensorrt_module'] = str(getattr(trt, '__file__', ''))
            report['tensorrt_version_match'] = tensorrt_versions_match(
                report['tensorrt'], report['host_tensorrt'])
            if report['jetson'] and report['tensorrt_version_match'] is False:
                report['formats']['engine'] = {
                    'available': False, 'argument': 'engine',
                    'reason': 'JetPack TensorRT ' + str(report['host_tensorrt']) +
                              ' differs from Python TensorRT ' + str(report['tensorrt']) +
                              '; select the matching SenSoRTC environment.'}
            else:
                report["formats"]["engine"] = {"available": True, "argument": "engine", "reason": "Ready"}
        except Exception as exc:
            report["errors"]["tensorrt"] = f"{type(exc).__name__}: {exc}"
            report["formats"]["engine"]["reason"] = (
                "TensorRT full Python builder is missing or cannot be loaded. " + str(exc)
            )
    return report


def parse_image_size(value):
    """Accept an integer or (height,width), returned in Ultralytics shape."""
    if isinstance(value, int) and not isinstance(value, bool):
        numbers = [value]
    elif isinstance(value, (list, tuple)):
        numbers = list(value)
    else:
        raw = str(value).strip().lower()
        raw = raw.strip("()[]")
        numbers = [x.strip() for x in re.split(r"\s*[x,;]\s*", raw) if x.strip()]
    if len(numbers) not in (1, 2):
        raise ValueError("Image size must be one integer (640) or height × width (e.g. 640x1088).")
    try:
        dims = [int(n) for n in numbers]
    except (TypeError, ValueError) as exc:
        raise ValueError("Image size must contain whole numbers, e.g. 640 or 640x1088.") from exc
    if any(d < 32 or d > 8192 for d in dims):
        raise ValueError("Image dimensions must be between 32 and 8192 pixels.")
    return dims[0] if len(dims) == 1 else tuple(dims)


def _positive_int(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer.")
    try:
        number = int(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be a positive integer.") from exc
    if number < 1:
        raise ValueError(f"{field} must be at least 1.")
    return number


def validate_export_settings(data, capabilities=None):
    """Validate a GUI or JSON export request, rejecting incompatible arguments."""
    config = dict(data)
    fmt = str(config.get("format", "onnx")).lower().strip()
    if fmt not in EXPORT_FORMATS:
        raise ValueError("Export format must be onnx, engine or litert.")
    if capabilities is not None:
        entry = capabilities.get("formats", {}).get(fmt, {})
        if not entry.get("available"):
            raise ValueError(f"{fmt.upper()} export is unavailable: {entry.get('reason', 'missing dependency')}")
        actual_format = entry.get("argument", fmt)
    else:
        actual_format = fmt
    source = str(config.get("model", "")).strip()
    if not source or not source.lower().endswith(".pt"):
        raise ValueError("Choose a trained YOLO .pt checkpoint or an official pretrained .pt name.")
    p = Path(source).expanduser()
    if p.is_absolute() or len(p.parts) > 1:
        if not p.is_file():
            raise ValueError(f"Model checkpoint does not exist: {source}")
        source = str(p.resolve())
    elif not p.is_file() and not re.fullmatch(r"(?:yolov(?:8|10)|yolo(?:11|12|26))[nsmlx]\.pt", source, re.I):
        raise ValueError(f"Model checkpoint does not exist: {source}")
    imgsz = parse_image_size(config.get("imgsz", 640))
    batch = _positive_int(config.get("batch", 1), "Batch size")
    precision = str(config.get("precision", "fp32")).lower().strip()
    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError("Precision must be FP32, FP16 or INT8.")
    backend = (capabilities or {}).get("formats", {}).get(fmt, {}).get("backend", "native")
    if fmt == "litert" and backend != "wsl" and (sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}):
        raise ValueError("LiteRT requires Linux x86-64 or an available Windows WSL2 export environment.")
    if fmt == "litert" and precision == "fp16":
        raise ValueError("LiteRT has no separate FP16 export; choose FP32 or INT8.")
    if config.get("dynamic") and fmt == "litert":
        raise ValueError("LiteRT export does not support dynamic shapes.")
    device = str(config.get("device", "auto")).lower().strip()
    if device not in ("auto", "cpu", "0", "cuda:0"):
        raise ValueError("Device must be auto, cpu, or GPU 0.")
    if fmt == "engine" and device == "cpu":
        raise ValueError("TensorRT engine building requires an NVIDIA GPU; choose Auto or GPU 0.")
    if fmt == "litert":
        device = "cpu"
    if fmt == "engine":
        device = "0"
    if precision == "int8":
        calibration = str(config.get("data", "")).strip()
        if not calibration or not Path(calibration).is_file():
            raise ValueError("INT8 quantization requires a representative calibration data.yaml file.")
        config["data"] = str(Path(calibration).expanduser().resolve())
    fraction = float(config.get("fraction", 1.0))
    if not math.isfinite(fraction) or not (0 < fraction <= 1):
        raise ValueError("Calibration fraction must be greater than 0 and no more than 1.")
    opset = str(config.get("opset", "")).strip()
    if opset:
        opset = _positive_int(opset, "ONNX opset")
        if fmt == "litert":
            raise ValueError("ONNX opset does not apply to LiteRT exports.")
    workspace = str(config.get("workspace", "")).strip()
    if workspace:
        try:
            workspace = float(workspace)
        except ValueError as exc:
            raise ValueError("TensorRT workspace must be a number of GiB.") from exc
        if not math.isfinite(workspace) or workspace <= 0:
            raise ValueError("TensorRT workspace must be greater than 0 GiB.")
    output = str(config.get("output", "")).strip()
    if output:
        output = str(Path(output).expanduser().resolve())
    model_name = normalize_export_model_name(config.get("model_name", ""), fmt)
    return {
        "model": source, "format": fmt, "ultralytics_format": actual_format,
        "imgsz": imgsz, "batch": batch, "precision": precision,
        "dynamic": bool(config.get("dynamic", False)),
        "simplify": bool(config.get("simplify", True)),
        "nms": bool(config.get("nms", False)), "device": device,
        "data": config.get("data", ""), "fraction": fraction,
        "workspace": workspace, "opset": opset,
        "output": output, "backend": backend,
        "model_name": model_name,
        "wsl_distribution": str(config.get("wsl_distribution", "")).strip(),
        "export_python": str(config.get("export_python", "")).strip(),
    }


def build_export_kwargs(config, supports_quantize=True):
    """Pass only format-appropriate Ultralytics arguments to YOLO.export()."""
    fmt = config["format"]
    result = {"format": config.get("ultralytics_format", fmt),
              "imgsz": config["imgsz"], "batch": config["batch"],
              "device": config["device"] if config["device"] != "auto" else None}
    if fmt in ("onnx", "engine"):
        result.update(dynamic=config["dynamic"], simplify=config["simplify"], nms=config["nms"])
        if config["opset"]:
            result["opset"] = config["opset"]
    if fmt == "engine" and config["workspace"]:
        result["workspace"] = config["workspace"]
    if config["precision"] != "fp32":
        if supports_quantize:
            result["quantize"] = 16 if config["precision"] == "fp16" else 8
        else:
            result["half"] = config["precision"] == "fp16"
            result["int8"] = config["precision"] == "int8"
    if config["precision"] == "int8":
        result["data"] = config["data"]
        result["fraction"] = config["fraction"]
    return {k: v for k, v in result.items() if v is not None}


def run_export_job(config, model_factory=None):
    """Export using native YOLO.export, keeping source checkpoints untouched."""
    # Must happen before importing Ultralytics: prevents automatic pip/apt
    # updates, including automatic replacement of JetPack's TensorRT package.
    os.environ['YOLO_AUTOINSTALL'] = 'False'
    real_export = model_factory is None
    if model_factory is None:
        from ultralytics import YOLO
        model_factory = YOLO
        # Never let an export implicitly install TensorRT after the user has
        # declined the GUI prompt. Recheck inside the actual worker process.
        capabilities = probe_export_environment()
    else:
        capabilities = None  # Fake model factories in offline tests.
    cfg = validate_export_settings(config, capabilities=capabilities)
    if capabilities is None and cfg["format"] == "litert":
        # A mock worker (or an already-probed request) may target the legacy
        # TFLite exporter. Only the known-safe compatibility alias is accepted.
        if config.get("ultralytics_format") in ("litert", "tflite"):
            cfg["ultralytics_format"] = config["ultralytics_format"]
    root = Path(cfg["output"] or Path.cwd() / "logs" / "exports")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_folder = root / f"export_{stamp}_{cfg['format']}"
    run_folder.mkdir(parents=True, exist_ok=False)
    source = Path(cfg["model"])
    if source.is_file():
        checkpoint = source
    else:
        print("Resolving pretrained model weights...", flush=True)
        original_model = model_factory(cfg["model"])
        path = getattr(original_model, "ckpt_path", None) or getattr(original_model, "pt_path", None)
        if not path or not Path(path).is_file():
            raise RuntimeError("Could not locate the downloaded pretrained checkpoint; select a local .pt file.")
        checkpoint = Path(path)
        del original_model
    import shutil
    working_input = run_folder / checkpoint.name
    shutil.copy2(checkpoint, working_input)
    print("STUDIO_EXPORT_STATUS=Preparing native Ultralytics exporter", flush=True)
    try:
        model = model_factory(str(working_input))
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            supports_quantize = "quantize" in DEFAULT_CFG_DICT
        except (ImportError, AttributeError):
            supports_quantize = False
        kwargs = build_export_kwargs(cfg, supports_quantize=supports_quantize)
        print("STUDIO_EXPORT_STATUS=Converting " + cfg["format"].upper() + " model", flush=True)
        exported = model.export(**kwargs)
        if exported is None:
            raise RuntimeError("Ultralytics did not return an exported model path.")
        result_path = Path(str(exported)).resolve()
        if not result_path.exists():
            raise RuntimeError(f"Ultralytics reported an output that does not exist: {result_path}")
        # Exporter behaviour differs across older releases; always put the
        # result in this job folder, without modifying the original .pt.
        if result_path != run_folder and run_folder not in result_path.parents:
            destination = run_folder / result_path.name
            if result_path.is_dir():
                shutil.copytree(result_path, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(result_path, destination)
            result_path = destination
        # Ultralytics chooses its own output name; apply the user's name only
        # *after* conversion and *before* TensorRT inference verification.
        # Never rename directories or overwrite an existing export artifact.
        requested = requested_export_filename(cfg["model_name"], cfg["format"])
        if requested:
            if not result_path.is_file():
                raise RuntimeError("Cannot rename this export: Ultralytics returned a directory, not a model file.")
            destination = run_folder / requested
            if result_path != destination:
                if destination.exists():
                    raise FileExistsError("Custom export filename already exists: " + str(destination))
                result_path.rename(destination)
                result_path = destination
        verification = {'verified': None, 'status': 'not_applicable'}
        if cfg['format'] == 'engine':
            if real_export:
                verification = validate_engine_with_selected_runtime(result_path, cfg['imgsz'])
            else:
                verification = {'verified': None, 'status': 'mock_export_not_verified'}
        metadata = {"source_model": str(checkpoint.resolve()), "exported_model": str(result_path),
                    "output_directory": str(run_folder.resolve()),
                    "settings": cfg, "ultralytics_arguments": kwargs,
                    "build_python": sys.executable,
                    "tensorrt_verification": verification}
        (run_folder / "export_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if verification.get('verified') is False:
            raise RuntimeError('TensorRT engine was exported but deployment validation failed: '
                               + verification.get('error', 'unknown failure')
                               + '. The unverified engine is kept for diagnostics; do not deploy it.')
        print("STUDIO_EXPORT_RESULT=" + json.dumps(metadata), flush=True)
        return metadata
    finally:
        # Delete only the private source *copy*, never the actual model.
        if working_input.is_file():
            working_input.unlink()


def validate_engine_with_selected_runtime(engine_path, imgsz, timeout=240):
    """Run a real synthetic inference in an isolated child of this interpreter.

    Child isolation prevents a broken/segfaulting TensorRT runtime from taking
    down the exporter before we can record its verification result.
    """
    env = dict(os.environ, YOLO_AUTOINSTALL='False')
    script = Path(__file__).with_name('verify_tensorrt_engine.py')
    try:
        result = subprocess.run([sys.executable, '-u', str(script), str(engine_path),
                                 '--imgsz', json.dumps(imgsz)],
                                cwd=str(script.parent), env=env, capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'verified': False, 'error': f'Could not run verification: {exc}'}
    record = next((line.split('=', 1)[1] for line in result.stdout.splitlines()
                   if line.startswith('STUDIO_ENGINE_VERIFICATION=')), None)
    if record:
        try:
            payload = json.loads(record)
        except ValueError:
            payload = {'verified': False, 'error': 'Malformed verifier result'}
    else:
        payload = {'verified': False,
                   'error': (result.stderr or result.stdout or 'Verifier returned no result')[-1500:]}
    if result.returncode != 0:
        payload['verified'] = False
    print('STUDIO_EXPORT_STATUS=' + ('TensorRT engine verified by inference' if payload.get('verified')
                                    else 'TensorRT engine FAILED deployment verification'), flush=True)
    return payload
