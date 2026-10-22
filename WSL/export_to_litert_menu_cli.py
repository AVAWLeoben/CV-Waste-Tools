#!/usr/bin/env python3
"""Export a YOLO .pt model to LiteRT and name the resulting .tflite by its settings."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_nms(value: str) -> bool | None:
    value = value.strip().lower()
    if value in {"none", "null", "auto"}:
        return None
    return parse_bool(value)


def parse_imgsz(value: str) -> int | list[int]:
    text = value.strip().lower().replace("x", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "imgsz must be an integer such as 640, or two integers such as 640,384"
        ) from exc
    if len(values) == 1 and values[0] > 0:
        return values[0]
    if len(values) == 2 and all(v > 0 for v in values):
        return values
    raise argparse.ArgumentTypeError(
        "imgsz must be one positive integer or two positive comma-separated integers"
    )


def make_output_name(model_path: Path, imgsz: int | list[int], nms: bool | None,
                     int8: bool, half: bool) -> str:
    size = str(imgsz) if isinstance(imgsz, int) else "x".join(map(str, imgsz))
    nms_text = "none" if nms is None else str(nms).lower()
    return (
        f"{model_path.stem}_img{size}_nms-{nms_text}"
        f"_int8-{str(int8).lower()}_half-{str(half).lower()}.tflite"
    )


def unused_output_path(target: Path) -> Path:
    """Never overwrite an earlier export with the same settings."""
    if not target.exists():
        return target
    index = 2
    while True:
        candidate = target.with_name(f"{target.stem}_{index}{target.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an Ultralytics YOLO model to LiteRT.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=parse_imgsz, default=640)
    parser.add_argument("--nms", type=parse_nms, default=None)
    parser.add_argument("--int8", type=parse_bool, default=False)
    parser.add_argument("--half", type=parse_bool, default=False)
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        print(f"\nERROR: Model not found: {model_path}")
        return 2
    if args.int8 and args.half:
        print("\nERROR: INT8 and HALF cannot both be enabled.")
        return 2

    export_args: dict[str, Any] = {
        "format": "litert",
        "imgsz": args.imgsz,
        "nms": args.nms,
    }
    if args.int8:
        export_args["int8"] = True
    if args.half:
        export_args["half"] = True

    target = model_path.with_name(make_output_name(
        model_path, args.imgsz, args.nms, args.int8, args.half
    ))
    print("\n" + "=" * 60)
    print("Ultralytics LiteRT export")
    print("=" * 60)
    print(f"Model  : {model_path}")
    for key, value in export_args.items():
        print(f"{key:7}: {value}")
    print(f"Naming : {target.name}")
    print("=" * 60 + "\n")

    model = YOLO(str(model_path))
    result = model.export(**export_args)

    # Ultralytics returns the actual exported .tflite file (rather than a
    # predictable name); use that return value to handle library changes.
    exported = Path(result).expanduser().resolve()
    if not exported.is_file() or exported.suffix.lower() != ".tflite":
        print(f"\nERROR: Expected an exported .tflite file, got: {exported}")
        return 1

    final_path = unused_output_path(target)
    exported.rename(final_path)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print(f"Output: {final_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
