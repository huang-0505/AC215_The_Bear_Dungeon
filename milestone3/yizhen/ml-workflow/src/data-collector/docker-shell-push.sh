#!/bin/bash
set -e

# ======== 基本变量设置 ========
export IMAGE_NAME="dnd-data-collector"
export PROJECT_ID="ac215"
export REGION="us-central1"
export REPO_NAME="ml-workflow"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"

echo "Pushing image to Artifact Registry: $IMAGE_URI"

# ======== Multi-arch builder setup ========
if docker buildx inspect multi-arch >/dev/null 2>&1; then
    echo "Removing existing multi-arch builder..."
    docker buildx rm multi-arch
fi

echo "Creating new multi-arch builder..."
docker buildx create --driver-opt network=host --use --name multi-arch

# ======== Build & Push ========
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  -t $IMAGE_URI \
  -f Dockerfile .

echo "✅ Build complete and pushed to Artifact Registry: $IMAGE_URI"
