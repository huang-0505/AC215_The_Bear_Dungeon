#!/bin/bash
set -e

echo "==============================================="
echo "🚀 Building & Launching DnD Workflow Container"
echo "==============================================="

# ----- Configuration -----
export IMAGE_NAME="dnd-workflow"
export BASE_DIR=$(pwd)
export SECRETS_DIR="$HOME/Desktop/secrets"

export GCP_PROJECT="even-turbine-471117-u0"
export GCP_REGION="us-central1"
export GCS_BUCKET_NAME="ac215-ml-workflow-central1"
export GCS_PACKAGE_URI="gs://ac215-ml-workflow-central1/model-training"
export GCS_SERVICE_ACCOUNT="ml-workflow@even-turbine-471117-u0.iam.gserviceaccount.com"

echo "🧩 Project: $GCP_PROJECT"
echo "📍 Region:  $GCP_REGION"
echo "🪣 Bucket:  $GCS_BUCKET_NAME"
echo "🔐 Service Account: $GCS_SERVICE_ACCOUNT"
echo "-----------------------------------------------"

# ----- Build Docker image -----
echo "🐳 Building Docker image..."
docker build --platform=linux/amd64 -t $IMAGE_NAME -f Dockerfile .

# ----- Run Docker container -----
echo "🚀 Starting container..."

docker run --rm -it \
    --name $IMAGE_NAME \
    -v "$BASE_DIR":/app \
    -v "$SECRETS_DIR":/secrets \
    -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ml-workflow.json \
    -e GCP_PROJECT="$GCP_PROJECT" \
    -e GCP_REGION="$GCP_REGION" \
    -e GCS_BUCKET_NAME="$GCS_BUCKET_NAME" \
    -e GCS_PACKAGE_URI="$GCS_PACKAGE_URI" \
    -e GCS_SERVICE_ACCOUNT="$GCS_SERVICE_ACCOUNT" \
    $IMAGE_NAME
