#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# ------------------------------------------------------------------------------
# QLoRA Fine-tuning helper script
# Builds and runs the fine-tuning container locally for debugging/training.
# ------------------------------------------------------------------------------

# (Optional) Load environment variables from a local config file if it exists.
# This file can define paths such as BASE_DIR or CACHE_DIR.
if [ -f ../env.dev ]; then
  source ../env.dev
  echo "Loaded environment variables from ../env.dev"
fi

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Image name for the container
export IMAGE_NAME="finetuning-qlora"

# Default directories (can be overridden by env.dev)
BASE_DIR="${BASE_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$BASE_DIR/data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$BASE_DIR/outputs}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# Create data/output folders if they don’t exist
mkdir -p "$DATA_DIR" "$OUTPUTS_DIR"

# ------------------------------------------------------------------------------
# Build Docker image
# ------------------------------------------------------------------------------

echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" -f Dockerfile .

# ------------------------------------------------------------------------------
# Run container interactively with GPU support and mounted volumes
# ------------------------------------------------------------------------------

echo "Running container: $IMAGE_NAME"

docker run --rm -it --gpus all \
  --name "$IMAGE_NAME" \
  -v "$DATA_DIR":/data \
  -v "$OUTPUTS_DIR":/outputs \
  -v "$HF_CACHE_DIR":/opt/hf \
  "$IMAGE_NAME" bash
