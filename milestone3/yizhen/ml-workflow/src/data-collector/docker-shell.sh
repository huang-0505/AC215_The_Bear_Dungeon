#!/bin/bash
set -e 


export IMAGE_NAME="dnd-data-collector"
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../../../persistent-folder/
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCP_PROJECT="ac215"
export GCS_BUCKET_NAME="ac215-ml-workflow"

docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .


docker run --rm --name $IMAGE_NAME -ti \
  -v "$BASE_DIR":/app \
  -v "$SECRETS_DIR":/secrets \
  -v "$PERSISTENT_DIR":/persistent \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ml-workflow.json \
  -e GCP_PROJECT=$GCP_PROJECT \
  -e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
  $IMAGE_NAME \
  --source_uri gs://dnd-master-dataset/dnd-narrator-finetune-dataset/ \
  --target_bucket ac215-ml-workflow
