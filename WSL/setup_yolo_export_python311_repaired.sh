#!/usr/bin/env bash
set -Eeuo pipefail

# LiteRT exporter setup. Python 3.11 is mandatory; never use system Python 3.14.
ENV_DIR="$HOME/yolo_export"
PY_REQUIRED="3.11"
UV_BIN="$HOME/.local/bin/uv"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
trap 'echo "Setup failed on line $LINENO." >&2' ERR

version_ok() {
    [ -x "$ENV_DIR/bin/python" ] && \
      "$ENV_DIR/bin/python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))' >/dev/null 2>&1
}

packages_ok() {
    version_ok && "$ENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version
import pip
import torch
import ultralytics
import tensorflow
import ai_edge_litert
import litert_torch

major, minor = (int(n) for n in version('litert-torch').split('.')[:2])
assert (major, minor) >= (0, 9), 'litert-torch >=0.9.0 is required'
PY
}

echo "Checking LiteRT export environment (Python $PY_REQUIRED)..."
if packages_ok; then
    echo 'Python 3.11, pip and all LiteRT export dependencies are available.'
    exit 0
fi

# uv installs managed CPython 3.11 without changing the system Python.
if [ ! -x "$UV_BIN" ]; then
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    else
        echo 'Installing uv...'
        if ! command -v curl >/dev/null 2>&1; then
            command -v apt-get >/dev/null 2>&1 || fail 'curl is required to install uv.'
            sudo apt-get update
            sudo apt-get install -y curl ca-certificates
        fi
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
        [ -x "$UV_BIN" ] || fail 'uv installation failed.'
    fi
fi

if ! version_ok; then
    echo "Creating Python $PY_REQUIRED environment..."
    "$UV_BIN" python install "$PY_REQUIRED"
    if [ -e "$ENV_DIR" ]; then
        echo "Replacing incomplete or incompatible environment: $ENV_DIR"
        rm -rf -- "$ENV_DIR"
    fi
    "$UV_BIN" venv --python "$PY_REQUIRED" --managed-python "$ENV_DIR"
fi

version_ok || fail 'Environment does not have Python 3.11.'

# uv-managed venvs normally omit pip. Ultralytics invokes `python -m pip`
# for dependency auto-installation, so explicitly install pip into the venv.
echo 'Installing/repairing Python packages (including pip and litert-torch)...'
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
    'pip>=24' \
    'torch' \
    'torchvision' \
    'ultralytics' \
    'tensorflow' \
    'ai-edge-litert>=2.1.4' \
    'litert-torch>=0.9.0'

if ! packages_ok; then
    echo 'Installed packages did not pass their import checks.' >&2
    "$ENV_DIR/bin/python" -m pip --version || true
    "$ENV_DIR/bin/python" -c 'import litert_torch; print("litert_torch OK")' || true
    fail 'LiteRT export dependencies are missing or incompatible; see messages above.'
fi

echo "Environment ready: $($ENV_DIR/bin/python --version)"
"$ENV_DIR/bin/python" -m pip --version
"$ENV_DIR/bin/python" -c 'from importlib.metadata import version; print("litert-torch", version("litert-torch"))'
