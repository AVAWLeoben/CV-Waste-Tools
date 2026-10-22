#!/usr/bin/env bash
set -Eeuo pipefail

# AVAW LiteRT WSL setup. Always uses Python 3.11, never Python 3.14.
ENV_DIR="$HOME/yolo_export"
PY_REQUIRED='3.11'
UV_BIN="$HOME/.local/bin/uv"

fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "Setup failed at line %s (exit code %s).\n" "$LINENO" "$?" >&2' ERR

python_ok() {
    [[ -x "$ENV_DIR/bin/python" ]] && \
        "$ENV_DIR/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' >/dev/null 2>&1
}

# Print the *actual* cause of a failed import or package-version check.
# ai-edge-litert is checked by installed distribution metadata; it is not
# guaranteed to expose an importable top-level `ai_edge_litert` module.
check_packages() {
    "$ENV_DIR/bin/python" - <<'PY'
import importlib
import re
import sys
from importlib.metadata import PackageNotFoundError, version

checks = [
    ("pip", "pip"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
    ("tensorflow", "tensorflow"),
    ("litert_torch", "litert_torch"),
]
failed = False
for label, module in checks:
    try:
        importlib.import_module(module)
        print(f"  OK: {label}", flush=True)
    except Exception as exc:
        print(f"  FAILED: {label}: {type(exc).__name__}: {exc}", flush=True)
        failed = True

for dist, minimum in (("litert-torch", (0, 9)), ("ai-edge-litert", (2, 1))):
    try:
        installed = version(dist)
        match = re.match(r"^(\d+)\.(\d+)", installed)
        if not match or tuple(map(int, match.groups())) < minimum:
            raise ValueError(f"requires >= {minimum[0]}.{minimum[1]} (found {installed})")
        print(f"  OK: {dist} {installed}", flush=True)
    except (PackageNotFoundError, ValueError) as exc:
        print(f"  FAILED: {dist}: {exc}", flush=True)
        failed = True

sys.exit(1 if failed else 0)
PY
}

# Fast launch check: query package metadata without importing PyTorch or
# TensorFlow. Only perform the full, slower import validation after a repair.
check_packages_fast() {
    "$ENV_DIR/bin/python" - <<'PYFAST'
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import re
import sys

required = (
    'pip', 'torch', 'torchvision', 'ultralytics', 'tensorflow',
    'litert-torch', 'ai-edge-litert',
)
minimums = {'litert-torch': (0, 9), 'ai-edge-litert': (2, 1)}
for package in required:
    try:
        installed = version(package)
        if package in minimums:
            match = re.match(r'^(\d+)\.(\d+)', installed)
            if match is None or tuple(map(int, match.groups())) < minimums[package]:
                sys.exit(1)
    except PackageNotFoundError:
        sys.exit(1)

if importlib.util.find_spec('pip') is None:
    sys.exit(1)
PYFAST
}

echo "Checking LiteRT environment: $ENV_DIR (Python $PY_REQUIRED)"
if python_ok; then
    if check_packages_fast; then
        echo 'LiteRT export environment is ready (quick check passed).'
        exit 0
    fi
    echo 'A dependency is missing or incompatible. Repairing environment...'
else
    echo 'Python 3.11 environment is missing or incompatible.'
fi

if [[ ! -x "$UV_BIN" ]]; then
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
        [[ -x "$UV_BIN" ]] || fail 'uv installation failed.'
    fi
fi

if ! python_ok; then
    echo "Installing managed Python $PY_REQUIRED (does not modify system Python)..."
    "$UV_BIN" python install "$PY_REQUIRED"
    if [[ -e "$ENV_DIR" ]]; then
        echo "Replacing incompatible or incomplete environment: $ENV_DIR"
        rm -rf -- "$ENV_DIR"
    fi
    "$UV_BIN" venv --python "$PY_REQUIRED" --managed-python "$ENV_DIR"
fi
python_ok || fail 'Failed to create a Python 3.11 environment.'

# uv venvs may not include pip. Ultralytics uses `python -m pip` when
# attempting to install export dependencies, so install pip explicitly.
echo 'Installing or repairing LiteRT packages...'
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
    'pip>=24' \
    torch torchvision ultralytics tensorflow \
    'ai-edge-litert>=2.1.4' \
    'litert-torch>=0.9.0'

echo 'Verifying installed packages...'
check_packages || fail 'Package verification failed. See the FAILED line(s) above for the exact cause.'
"$ENV_DIR/bin/python" -m pip --version
echo 'LiteRT export environment is ready.'
