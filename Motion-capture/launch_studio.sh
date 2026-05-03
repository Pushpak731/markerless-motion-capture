#!/bin/bash
# 3D Avatar Studio Launcher
# This script activates the virtual environment and starts the native 3D studio.

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  💎 MoCap 3D Avatar Studio - Launching"
echo "=========================================="

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "[error] Virtual environment (.venv) not found."
    echo "Please run: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

# Launch using the venv python directly
./.venv/bin/python tools/local_3d_studio.py
