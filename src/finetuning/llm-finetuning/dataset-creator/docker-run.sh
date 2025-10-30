#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Read the settings file
source ../env.dev

export IMAGE_NAME="llm-dataset-creator"

# Run Container (without building)
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
$IMAGE_NAME
