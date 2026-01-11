#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/Llama-3.1-8B-Instruct-CoreML/llama_3.1_coreml.mlpackage}"
VENV_DIR="${2:-.venv}"

echo "==> Checking macOS version"
if ! command -v sw_vers >/dev/null 2>&1; then
  echo "Error: sw_vers not found. This script is for macOS only."
  exit 1
fi
PRODUCT_VERSION=$(sw_vers -productVersion)
echo "macOS version: ${PRODUCT_VERSION}"

# Optional: enforce minimum version, adjust as you like
MAJOR=$(echo "$PRODUCT_VERSION" | cut -d. -f1)
if [ "$MAJOR" -lt 15 ]; then
  echo "Error: need macOS 15+ (or 26+ equivalent) for Core ML state API."
  exit 1
fi

echo "==> Checking Python"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found. Install Python 3.10+ first."
  exit 1
fi
PYVER=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
echo "Python: ${PYVER}"

echo "==> Creating virtualenv at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip and installing coremltools"
python -m pip install --upgrade pip
python -m pip install --upgrade coremltools

echo "==> Verifying coremltools and model state support"
python - <<EOF
import os
import coremltools as ct

print("coremltools version:", ct.__version__)

model_path = "${MODEL_PATH}"
if not os.path.exists(model_path):
    raise SystemExit(f"ERROR: Model not found at {model_path}. Put your .mlpackage there or pass a custom path.")

mlmodel = ct.models.MLModel(model_path)
print("has make_state():", hasattr(mlmodel, "make_state"))

if hasattr(mlmodel, "make_state"):
    try:
        state = mlmodel.make_state()
        print("make_state() call: OK")
        # Some models expose dict-like state; others are opaque
        if isinstance(state, dict):
            print("state keys:", list(state.keys()))
        else:
            print("state is non-dict (opaque) object:", type(state))
    except Exception as e:
        print("make_state() raised:", repr(e))
else:
    print("WARNING: model does not support make_state(). You may need a newer .mlpackage.")
EOF

echo "==> Setup complete."
echo "Activate env with:  source ${VENV_DIR}/bin/activate"
