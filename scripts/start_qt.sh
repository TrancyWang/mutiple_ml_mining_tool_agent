#!/usr/bin/env bash
set -euo pipefail

echo "启动 Academic Agent PySide6 对话界面"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/trancy_tool/bin/python}"
exec "$PYTHON_BIN" "$PROJECT_DIR/qt_client.py"
