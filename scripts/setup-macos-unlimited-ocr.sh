#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Apple Silicon arm64 is required for the macOS Unlimited-OCR MPS route."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  else
    PYTHON_BIN="python3"
  fi
fi

VENV_DIR="${UNLIMITED_OCR_MACOS_VENV:-.venv-unlimited-ocr-macos}"

echo "Using Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating Unlimited-OCR virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Using existing Unlimited-OCR virtual environment: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-macos-unlimited-ocr.txt

python - <<'PY'
import importlib.util
import torch
import transformers

required = [
    "fastapi",
    "fitz",
    "httpx",
    "multipart",
    "PIL",
    "psutil",
    "torch",
    "torchvision",
    "transformers",
    "uvicorn",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Unlimited-OCR dependencies: {', '.join(missing)}")

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("mps available", torch.backends.mps.is_available())
PY

echo "macOS Unlimited-OCR setup complete."
echo "Start with: PANDOCR_ENABLE_UNLIMITED_OCR=1 bash scripts/start-macos.sh"
