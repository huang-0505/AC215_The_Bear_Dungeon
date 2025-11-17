#!/bin/bash
set -e

echo "==============================================="
echo "🚀 Workflow container started"
echo "==============================================="

echo "Python: $(python --version)"
echo "UV: $(uv --version || echo 'uv not found')"
echo "Workdir: $(pwd)"
echo "Args: $@"
echo "-----------------------------------------------"

# Activate virtual environment
if [ -f "/.venv/bin/activate" ]; then
  echo "🔹 Activating virtual environment..."
  . /.venv/bin/activate
else
  echo "⚠️  No virtual environment found. Using system Python."
fi


if [ $# -eq 0 ]; then
  echo "💡 No args provided → running: uv run python cli.py"
  exec uv run python cli.py
fi

echo "🔹 Running CLI: uv run python cli.py $@"
exec uv run python cli.py "$@"
