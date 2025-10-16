#!/usr/bin/env bash
set -e

echo "========================================"
echo "🚀 Container is running — QLoRA Fine-tuning Environment"
echo "========================================"

echo "Architecture: $(uname -m)"
echo "Python version: $(python --version | awk '{print $2}')"
echo "UV version: $(uv --version)"
echo "Working directory: $(pwd)"
echo "----------------------------------------"

# Activate virtual environment
echo "Activating virtual environment..."
source /.venv/bin/activate
echo "Environment ready!"
echo "----------------------------------------"

# If arguments are provided, run them (for debugging or custom commands)
if [ $# -gt 0 ]; then
  echo "Executing custom command: $@"
  exec "$@"
fi

# Otherwise, run default fine-tuning command
echo "No custom command detected — starting QLoRA training..."
exec python cli.py \
  --model_name "${MODEL_NAME}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LR}"
