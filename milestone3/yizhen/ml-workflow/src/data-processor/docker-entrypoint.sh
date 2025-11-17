#!/bin/bash
set -e

echo "============================================"
echo "🚀 DnD Data Processor Container Started"
echo "============================================"
echo "Architecture: $(uname -m)"
echo "Python version: $(python --version)"
echo "UV version: $(uv --version)"
echo "GCP_PROJECT=${GCP_PROJECT}"
echo "GCS_BUCKET_NAME=${GCS_BUCKET_NAME}"
echo "GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}"
echo "--------------------------------------------"

echo "🔧 Activating virtual environment..."
source /.venv/bin/activate

# If no args → default: run cli
if [[ $# -eq 0 ]]; then
    echo "💡 No arguments provided → running default: cli.py"
    uv run python cli.py
    exit 0
fi

# If args provided → run them directly through python
echo "▶️ Running: uv run python $@"
uv run python "$@"
