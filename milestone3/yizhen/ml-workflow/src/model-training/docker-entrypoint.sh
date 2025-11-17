#!/bin/bash
set -e

echo "============================================"
echo "🚀 DnD Model-Training Container Started"
echo "============================================"
echo "Architecture: $(uname -m)"
echo "Python version: $(python --version)"
echo "UV version: $(uv --version)"
echo "GCP_PROJECT=${GCP_PROJECT}"
echo "GCS_BUCKET_NAME=${GCS_BUCKET_NAME}"
echo "GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}"
echo "--------------------------------------------"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [ -d "/.venv" ]; then
  source /.venv/bin/activate
fi

# Default to "tune"
if [ $# -eq 0 ]; then
  echo "💡 No arguments provided → default: tune"
  set -- tune
fi

echo "▶️ Running CLI with args: $@"

# Always run CLI. Never re-interpret arguments.
exec uv run python /app/cli.py "$@"
