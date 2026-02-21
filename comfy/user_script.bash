#!/usr/bin/env bash
set -euo pipefail

VENV_PY="/comfy/mnt/venv/bin/python"
MARKER_FILE="/comfy/mnt/.deps_installed_v1"

if [ ! -x "$VENV_PY" ]; then
  echo "venv python not ready yet, skipping dependency install"
  exit 0
fi

if [ -f "$MARKER_FILE" ]; then
  echo "custom dependencies already installed"
  exit 0
fi

"$VENV_PY" -m pip install -U uv
"$VENV_PY" -m uv pip install "transformers==4.57.3" aiofiles "python-socketio>=5"
"$VENV_PY" -c "import transformers, aiofiles, socketio; print(transformers.__version__); print(aiofiles.__version__); print(socketio.__version__)"

touch "$MARKER_FILE"
echo "custom dependencies installed"
