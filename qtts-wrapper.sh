#!/bin/bash
# qtts wrapper script - activates virtual environment and runs qtts.py
# This script should be symlinked to /usr/local/bin/qtts for system-wide access

# Get the real path of this script (follows symlinks)
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || greadlink -f "${BASH_SOURCE[0]}" 2>/dev/null || realpath "${BASH_SOURCE[0]}" 2>/dev/null)"

# If readlink/realpath not available, use Python to resolve the symlink
if [ -z "$SCRIPT_PATH" ] || [ ! -f "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="$(python3 -c "import os; print(os.path.realpath('${BASH_SOURCE[0]}'))")"
fi

# Get the directory where the actual script is located
QTTS_DIR="$( cd "$( dirname "$SCRIPT_PATH" )" && pwd )"
VENV_PYTHON="$QTTS_DIR/venv/bin/python3"
QTTS_SCRIPT="$QTTS_DIR/qtts.py"

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at $VENV_PYTHON" >&2
    echo "Please run: cd $QTTS_DIR && python3.12 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# Check if qtts.py exists
if [ ! -f "$QTTS_SCRIPT" ]; then
    echo "Error: qtts.py not found at $QTTS_SCRIPT" >&2
    exit 1
fi

# Run qtts.py with the virtual environment's Python, passing all arguments
exec "$VENV_PYTHON" "$QTTS_SCRIPT" "$@"
