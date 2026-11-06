"""Post-export deployment validation in the selected Python/TensorRT runtime.

A successful verification requires both engine deserialization and one real
inference call, using Ultralytics' existing TensorRT inference backend. It is
intentionally not a substitute for validating predictions on real camera data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def verify_engine(engine_file: str | Path, imgsz=640) -> dict:
    os.environ['YOLO_AUTOINSTALL'] = 'False'
    engine_file = Path(engine_file).resolve()
    if not engine_file.is_file() or engine_file.suffix.lower() != '.engine':
        raise ValueError(f'Expected an existing .engine file: {engine_file}')
    import tensorrt as trt
    import torch
    import numpy as np
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available to the selected deployment interpreter.')
    dims = [int(imgsz)] * 2 if isinstance(imgsz, int) else [int(v) for v in imgsz]
    if len(dims) != 2 or any(v <= 0 for v in dims):
        raise ValueError('Invalid model image shape')
    # Model.predict uses the actual Ultralytics TensorRT loader: it handles the
    # Ultralytics metadata prefix before TRT engine deserialization correctly.
    model = YOLO(str(engine_file), task=None)
    sample = np.zeros((dims[0], dims[1], 3), dtype=np.uint8)
    predictions = model.predict(source=sample, imgsz=dims, batch=1, device=0, verbose=False)
    if not isinstance(predictions, list) or not predictions:
        raise RuntimeError('TensorRT inference did not return results.')
    return {'verified': True, 'test': 'engine_deserialization_and_synthetic_inference',
            'engine': str(engine_file), 'python': sys.executable,
            'tensorrt': str(trt.__version__), 'tensorrt_module': str(getattr(trt, '__file__', '')),
            'cuda': str(torch.version.cuda), 'gpu': torch.cuda.get_device_name(0),
            'image_shape': dims, 'results': len(predictions)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('engine')
    parser.add_argument('--imgsz', required=True, help='JSON integer or [height,width]')
    args = parser.parse_args(argv)
    try:
        result = verify_engine(args.engine, json.loads(args.imgsz))
        print('STUDIO_ENGINE_VERIFICATION=' + json.dumps(result), flush=True)
        return 0
    except Exception as exc:
        print('STUDIO_ENGINE_VERIFICATION=' + json.dumps({
            'verified': False, 'engine': args.engine, 'python': sys.executable,
            'error': f'{type(exc).__name__}: {exc}'}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
