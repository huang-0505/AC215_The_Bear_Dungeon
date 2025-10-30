#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Read the settings file
source ../env.dev

export IMAGE_NAME="hf-autotrainer"

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME  -f Dockerfile .

# Run Container
docker run --rm --gpus all --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_SERVICE_ACCOUNT=$GCP_SERVICE_ACCOUNT \
$IMAGE_NAME