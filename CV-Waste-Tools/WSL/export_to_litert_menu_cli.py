#!/usr/bin/env python3
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export an Ultralytics YOLO model to LiteRT."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=parse_imgsz, default=640)
    parser.add_argument("--nms", type=parse_nms, default=None)
    parser.add_argument("--int8", type=parse_bool, default=False)
    parser.add_argument("--half", type=parse_bool, default=False)
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()

    if not model_path.exists():
        print(f"\nERROR: Model not found: {model_path.resolve()}")
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

    print("\n" + "=" * 60)
    print("Ultralytics LiteRT export")
    print("=" * 60)
    print(f"Model : {model_path.resolve()}")
    for key, value in export_args.items():
        print(f"{key:6}: {value}")
    print("=" * 60 + "\n")

    model = YOLO(str(model_path))
    result = model.export(**export_args)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print(f"Output: {result}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
