#!/bin/bash
set -e

# ============================================================
# 🚀 DnD Data Processor - Local Docker Shell (DEBUG MODE)
# ============================================================

export IMAGE_NAME="dnd-processor"

export BASE_DIR=$(pwd)

# Secrets folder must contain: ml-workflow.json
export SECRETS_DIR=$(pwd)/../../../secrets/

# Optional persistent folder (not required but kept)
export PERSISTENT_DIR=$(pwd)/../../../persistent-folder/

export GCP_PROJECT="ac215"
export GCS_BUCKET_NAME="ac215-ml-workflow"

echo "============================================"
echo "Building Docker image for Data Processor..."
echo "Image Name: $IMAGE_NAME"
echo "Project: $GCP_PROJECT"
echo "Bucket: $GCS_BUCKET_NAME"
echo "============================================"

# ============================================================
# 🧱 Build image (use amd64 for Vertex AI / GCP compatibility)
# ============================================================
docker build --platform=linux/amd64 -t $IMAGE_NAME -f Dockerfile .

# ============================================================
# 🧩 Launch container (interactive)
# ============================================================
echo "============================================"
echo "Starting DnD Data Processor container (interactive)..."
echo "Use commands such as:"
echo "   python cli.py --process"
echo "============================================"

docker run --rm -it \
  -v "$BASE_DIR":/app \
  -v "$SECRETS_DIR":/app/secrets \
  -v "$PERSISTENT_DIR":/persistent \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/ml-workflow.json \
  -e GCP_PROJECT=$GCP_PROJECT \
  -e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
  $IMAGE_NAME \
  /bin/bash
