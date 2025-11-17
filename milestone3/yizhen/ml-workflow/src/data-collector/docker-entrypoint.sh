#!/bin/bash
set -e

echo "Starting DND Data Collector container..."
echo "Architecture: $(uname -m)"
echo "Python version: $(python --version)"
echo "UV version: $(uv --version)"
echo "Using GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}"

# Activate virtual environment
echo "Activating virtual environment..."
source /.venv/bin/activate

args="$@"
echo "Received args: $args"

# Default behavior: run collector
if [[ -z "$args" ]]; then
    echo "No arguments provided → running default: cli.py --run"
    uv run python cli.py --run
else
    echo "Running cli.py with provided arguments..."
    uv run python cli.py "$@"
fi

