#!/usr/bin/env bash
set -euo pipefail
ENV_DIR="$HOME/yolo_export"
if [ -x "$ENV_DIR/bin/python" ] && "$ENV_DIR/bin/python" -c 'import torch, ultralytics, tensorflow, ai_edge_litert' >/dev/null 2>&1; then
    echo "Existing YOLO export environment is ready."
    exit 0
fi
if ! command -v python3 >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip
fi
if [ ! -x "$ENV_DIR/bin/python" ]; then
    if ! python3 -m venv "$ENV_DIR"; then
        sudo apt-get update
        sudo apt-get install -y python3-venv python3-pip
        python3 -m venv "$ENV_DIR"
    fi
fi
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/python" -m pip install torch torchvision ultralytics tensorflow ai-edge-litert
"$ENV_DIR/bin/python" -c 'import torch, ultralytics, tensorflow, ai_edge_litert; print("Environment OK")'
