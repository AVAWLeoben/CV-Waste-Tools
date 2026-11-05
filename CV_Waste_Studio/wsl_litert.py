"""Optional Windows -> WSL2 LiteRT bridge for CV Waste Studio.

No automatic WSL installation: probing is read-only and environment changes are
only made after the operator explicitly opts in from Model Export.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WSL_SETUP_FILE = "setup_yolo_export.sh"
WSL_PYTHON = "$HOME/yolo_export/bin/python"
REQUIRED_PACKAGES = ("torch", "ultralytics", "litert-torch")


def decode_wsl_output(value):
    """WSL list output is often UTF-16LE, unlike --exec command output."""
    if isinstance(value, str):
        return value.replace("\x00", "").lstrip("\ufeff").strip()
    if not value:
        return ""
    if value.startswith((b"\xff\xfe", b"\xfe\xff")):
        return value.decode("utf-16", errors="replace").strip()
    if b"\x00" in value:
        return value.decode("utf-16-le", errors="replace").lstrip("\ufeff").strip()
    return value.decode("utf-8", errors="replace").lstrip("\ufeff").strip()


def _execute(args, timeout=30):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          stdin=subprocess.DEVNULL, timeout=timeout, check=False)


def _failure(stage, message, **extra):
    return dict(available=False, phase=stage, reason=message, distribution=None,
                distributions=[], **extra)


def _probe_distribution(distro, *, run, version=None):
    """Read-only probe of a single WSL distribution's *existing* exporter."""
    common = {"distribution": distro}
    if version == "1":
        return {**_failure("wsl1", f"{distro} uses WSL1. Run wsl --set-version {distro} 2."), **common}
    try:
        arch = run(["wsl.exe", "--distribution", distro, "--exec", "/bin/uname", "-m"], timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {**_failure("distro_not_ready", f"Could not start {distro}: {exc}"), **common}
    if arch.returncode:
        message = decode_wsl_output(arch.stderr or arch.stdout)
        return {**_failure("distro_not_ready", f"Could not start {distro}: {message[:250]}"), **common}
    if decode_wsl_output(arch.stdout).lower() not in ("x86_64", "amd64"):
        return {**_failure("unsupported_arch", f"{distro} is not Linux x86-64."), **common}

    # Modern Ultralytics LiteRT is Torch -> litert_torch, not TensorFlow export.
    # An already-working WSL exporter must not be rejected simply because the
    # legacy tensorflow distribution is absent. This also accepts Python 3.12,
    # which is known to work with Ultralytics 8.4.174 + litert_torch 0.9.4.
    code = r'''import importlib.metadata as metadata, json, sys
result = {"python": sys.version.split()[0], "executable": sys.executable,
          "missing": [], "optional_missing": [], "format": None}
for name in ("torch", "ultralytics", "litert-torch"):
    try:
        result[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        result["missing"].append(name)
for name in ("torchvision", "ai-edge-litert"):
    try:
        result[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        result["optional_missing"].append(name)
try:
    from ultralytics.engine.exporter import export_formats
    table = export_formats()
    if hasattr(table, "to_dict"):
        formats = [str(x).lower().strip() for x in table["Argument"].tolist()]
    else:
        formats = [str(x).lower().strip() for x in table.get("Argument", [])]
    result["format"] = "litert" if "litert" in formats else "tflite" if "tflite" in formats else None
except Exception as exc:
    result["import_error"] = type(exc).__name__ + ": " + str(exc)
print("STUDIO_WSL_PROBE=" + json.dumps(result), flush=True)'''
    bash = ('test -x "$HOME/yolo_export/bin/python" || exit 42; '
            'exec "$HOME/yolo_export/bin/python" -c "$1"')
    try:
        checked = run(["wsl.exe", "--distribution", distro, "--exec", "/bin/bash", "-c",
                       bash, "wsl-probe", code], timeout=105)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {**_failure("env_error", f"Could not inspect {distro}/~/yolo_export: {exc}"), **common}
    if checked.returncode == 42:
        return {**_failure("missing_env", f"{distro}: ~/yolo_export/bin/python was not found. "
                           "An existing exporter may be installed in a different distribution or environment."), **common}
    output = decode_wsl_output(checked.stdout)
    line = next((x.split("=", 1)[1] for x in output.splitlines()
                 if x.startswith("STUDIO_WSL_PROBE=")), None)
    if not line:
        error = decode_wsl_output(checked.stderr or checked.stdout)
        return {**_failure("env_error", f"{distro} export probe exited {checked.returncode}: {error[:500]}"), **common}
    try:
        info = json.loads(line)
    except ValueError:
        return {**_failure("env_error", f"{distro} returned invalid probe metadata."), **common}
    if checked.returncode:
        return {**_failure("env_error", f"{distro} export probe failed (exit {checked.returncode}): "
                           + decode_wsl_output(checked.stderr)[:350], packages=info), **common}
    if info.get("import_error"):
        return {**_failure("env_error", f"{distro}: Ultralytics exporter import failed: {info['import_error']}",
                           packages=info), **common}
    if not info.get("format"):
        return {**_failure("unsupported_format", f"{distro}: installed Ultralytics does not advertise "
                           "LiteRT or TFLite export.", packages=info), **common}
    if info.get("missing"):
        return {**_failure("missing_packages", f"{distro}: required export packages missing: "
                           + ", ".join(info["missing"]), packages=info), **common}
    warnings = list(info.get("optional_missing", []))
    reason = f"Ready: WSL2 {distro}, Python {info.get('python', 'unknown')}, {info['format']} exporter"
    if warnings:
        reason += " (optional package checks: " + ", ".join(warnings) + ")"
    return {**common, "available": True, "phase": "ready", "reason": reason,
            "argument": info["format"], "packages": info, "warnings": warnings}


def probe_wsl_litert(run=_execute):
    """Find a usable WSL2 exporter across registered distributions without modifying it."""
    if sys.platform != "win32":
        return _failure("non_windows", "WSL export is applicable only to Windows hosts.")
    try:
        listed = run(["wsl.exe", "--list", "--quiet"], timeout=16)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _failure("missing_wsl", f"WSL is unavailable: {exc}")
    if listed.returncode:
        error = decode_wsl_output(listed.stderr or listed.stdout)
        return _failure("missing_wsl", "WSL is not installed or enabled. " + error[:200])
    names = [s.strip().lstrip("*").strip() for s in decode_wsl_output(listed.stdout).splitlines() if s.strip()]
    if not names:
        return _failure("missing_distro", "WSL is installed, but no Linux distribution is registered.")
    versions = {}
    try:
        verbose = run(["wsl.exe", "--list", "--verbose"], timeout=14)
        if verbose.returncode == 0:
            for line in decode_wsl_output(verbose.stdout).splitlines():
                parts = line.lstrip().lstrip("*").split()
                if len(parts) >= 3 and parts[-1] in {"1", "2"}:
                    versions[" ".join(parts[:-2])] = parts[-1]
    except (OSError, subprocess.TimeoutExpired):
        pass
    attempts = []
    for distro in names:
        outcome = _probe_distribution(distro, run=run, version=versions.get(distro))
        attempts.append({"distribution": distro, "phase": outcome.get("phase"),
                         "reason": outcome.get("reason")})
        if outcome.get("available"):
            return {**outcome, "distributions": names, "attempts": attempts}
    # Prefer actionable details from a WSL2 environment over an unrelated
    # WSL1 distribution or the empty first/default distribution.
    priority = {"missing_packages": 0, "unsupported_format": 1, "env_error": 2,
                "missing_env": 3, "distro_not_ready": 4, "unsupported_arch": 5, "wsl1": 6}
    outcomes = []
    for attempt in attempts:
        outcomes.append(attempt)
    choice = min(outcomes, key=lambda a: priority.get(a["phase"], 99))
    return {"available": False, "phase": choice["phase"],
            "reason": choice["reason"], "distribution": choice["distribution"],
            "distributions": names, "attempts": attempts}


def windows_to_wsl(path, distro, run=_execute):
    """Convert paths through WSL rather than hardcoding /mnt/c/ or drive letters."""
    result = run(["wsl.exe", "--distribution", distro, "--exec", "wslpath", "-a", "-u", str(path)], timeout=15)
    if result.returncode:
        raise RuntimeError("Could not translate Windows path for WSL: " +
                           decode_wsl_output(result.stderr or result.stdout)[:250])
    value = decode_wsl_output(result.stdout).strip()
    if not value.startswith("/"):
        raise RuntimeError(f"WSL path conversion returned an invalid path: {value!r}")
    return value


def run_wsl_litert_export(config, *, probe=probe_wsl_litert, run=_execute, popen=subprocess.Popen):
    """Call YOLO.export() in a managed WSL2 Python, preserving the Windows .pt."""
    from model_export import validate_export_settings

    capabilities = probe()
    if not capabilities.get("available"):
        raise RuntimeError("LiteRT WSL environment is not ready: " + capabilities.get("reason", "unknown"))
    distro = config.get("wsl_distribution") or capabilities["distribution"]
    if distro not in capabilities["distributions"]:
        raise ValueError(f"WSL distribution is not registered: {distro}")
    # The probe validates the format inside the Linux env, independently of Windows.
    fmt_cap = {"formats": {"litert": {"available": True, "argument": capabilities["argument"], "backend": "wsl"}}}
    cfg = validate_export_settings(config, capabilities=fmt_cap)
    src = Path(cfg["model"])
    if not src.is_file():
        raise ValueError("WSL LiteRT export requires a local .pt file. Download pretrained weights first.")
    root = Path(cfg["output"] or Path.cwd() / "logs" / "exports")
    root.mkdir(parents=True, exist_ok=True)
    job_folder = root / ("export_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_litert")
    job_folder.mkdir(parents=False, exist_ok=False)
    copy = job_folder / src.name
    shutil.copy2(src, copy)
    print("STUDIO_EXPORT_WORKDIR=" + str(job_folder), flush=True)
    print("STUDIO_EXPORT_STATUS=Preparing Linux LiteRT export in WSL2", flush=True)
    try:
        wsl_copy = windows_to_wsl(copy, distro, run=run)
        wsl_output = windows_to_wsl(job_folder, distro, run=run)
        wsl_script = windows_to_wsl(Path(__file__).with_name("wsl_litert_export_worker.py"), distro, run=run)
        calib_wsl = ""
        if cfg["precision"] == "int8":
            calib_wsl = windows_to_wsl(cfg["data"], distro, run=run)
        job = {"model": wsl_copy, "output": wsl_output, "format": capabilities["argument"],
               "imgsz": cfg["imgsz"], "batch": cfg["batch"], "precision": cfg["precision"],
               "data": calib_wsl, "fraction": cfg["fraction"], "nms": cfg["nms"]}
        json_file = job_folder / "wsl_export_request.json"
        json_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
        wsl_request = windows_to_wsl(json_file, distro, run=run)
        # A fixed shell expression reads only positional arguments ($@). Paths,
        # spaces and non-ASCII characters are never interpolated as shell code.
        command = ["wsl.exe", "--distribution", distro, "--exec", "/bin/bash", "-c",
                   'exec "$HOME/yolo_export/bin/python" "$@"', "studio-wsl-export", wsl_script, wsl_request]
        print(f"STUDIO_EXPORT_STATUS=Exporting LiteRT in WSL2 ({distro})", flush=True)
        proc = popen(command, stdin=subprocess.DEVNULL)
        returncode = proc.wait()
        if returncode:
            raise RuntimeError(f"WSL LiteRT export failed (exit code {returncode}); see the export log.")
        wsl_result_path = job_folder / "wsl_export_result.json"
        if not wsl_result_path.is_file():
            raise RuntimeError("WSL reported success but no result metadata was saved.")
        result = json.loads(wsl_result_path.read_text(encoding="utf-8"))
        artifact = job_folder / result["filename"]
        if not artifact.is_file() or artifact.suffix.lower() != ".tflite":
            raise RuntimeError("WSL export did not produce the expected .tflite file.")
        metadata = {"source_model": str(src.resolve()), "exported_model": str(artifact.resolve()),
                    "output_directory": str(job_folder.resolve()), "settings": cfg,
                    "backend": "wsl", "wsl_distribution": distro,
                    "ultralytics_arguments": result.get("ultralytics_arguments", {})}
        (job_folder / "export_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("STUDIO_EXPORT_RESULT=" + json.dumps(metadata), flush=True)
        return metadata
    finally:
        # Never remove the user's original weights, only the temporary private copy.
        copy.unlink(missing_ok=True)
