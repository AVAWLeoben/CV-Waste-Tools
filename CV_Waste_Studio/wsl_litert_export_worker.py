"""Executed only inside the managed WSL2 Linux Python (not Windows Conda)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def _calibration_yaml(source, destination):
    """Resolve relative YAML paths against the *original* YAML directory.

    Windows-style absolute paths inside the YAML are not silently rewritten;
    their conversion requires a dataset-specific mapping and is rejected.
    """
    import yaml

    original = Path(source)
    doc = yaml.safe_load(original.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("Calibration YAML must be a mapping.")
    for key in ("path", "train", "val", "test"):
        text = doc.get(key)
        strings = [text] if isinstance(text, str) else text if isinstance(text, list) else []
        for item in strings:
            if isinstance(item, str) and len(item) > 2 and item[1:3] in (":\\", ":/"):
                raise ValueError("Calibration YAML contains a Windows path. Convert its path/train/val/test "
                                 "entries to WSL paths before INT8 export: " + item)
    doc["path"] = str(Path(doc.get("path") or original.parent).resolve()) if not doc.get("path") else (
        str((original.parent / doc["path"]).resolve()) if not Path(str(doc["path"])).is_absolute() else doc["path"])
    output = Path(destination)
    output.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return str(output)


def main(argv=None, model_factory=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    # WSL export is a conversion, not permission to modify the Linux venv.
    os.environ['YOLO_AUTOINSTALL'] = 'False'
    config = json.loads(args.request.read_text(encoding="utf-8"))
    model_path = Path(config["model"])
    if not model_path.is_file():
        raise FileNotFoundError("Staged .pt checkpoint not found in WSL: " + str(model_path))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "wsl_export.pid").write_text(str(os.getpid()), encoding="utf-8")
    try:
        if model_factory is None:
            from ultralytics import YOLO
            model_factory = YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        kwargs = {"format": config["format"], "imgsz": config["imgsz"],
                  "batch": config["batch"], "device": "cpu"}
        if config.get("nms"):
            kwargs["nms"] = True
        if config["precision"] == "int8":
            if not config.get("data"):
                raise ValueError("INT8 export requires representative calibration data.yaml.")
            kwargs["data"] = _calibration_yaml(config["data"], output / "calibration_wsl.yaml")
            kwargs["fraction"] = config["fraction"]
            if "quantize" in DEFAULT_CFG_DICT:
                kwargs["quantize"] = 8
            else:
                kwargs["int8"] = True
        print("STUDIO_EXPORT_STATUS=Running native Ultralytics YOLO.export() in WSL2", flush=True)
        result = model_factory(str(model_path)).export(**kwargs)
        exported = Path(str(result)).resolve()
        if not exported.is_file() or exported.suffix.lower() != ".tflite":
            raise RuntimeError("Ultralytics did not produce a .tflite model: " + str(exported))
        from model_export import requested_export_filename
        desired_name = requested_export_filename(config.get("model_name", ""), "litert")
        dest = output / (desired_name or (
            model_path.stem + "_" + str(config["imgsz"]).replace(" ", "").replace("[", "").replace("]", "").replace(",", "x") +
            ("_int8" if config["precision"] == "int8" else "_fp32") + ".tflite"))
        if dest != exported:
            if dest.exists():
                raise FileExistsError("Output already exists: " + str(dest))
            exported.rename(dest)
        summary = {"filename": dest.name, "ultralytics_arguments": kwargs}
        (output / "wsl_export_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("STUDIO_EXPORT_STATUS=LiteRT conversion finished", flush=True)
        return summary
    finally:
        (output / "wsl_export.pid").unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"WSL export error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
