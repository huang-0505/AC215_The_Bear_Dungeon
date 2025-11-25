#!/bin/bash

# docker-shell.sh
# Build and run the DnD Combat Simulator container interactively

set -e

# Configuration
export IMAGE_NAME="dnd-combat-simulator"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../secrets
export GCP_PROJECT="even-turbine-471117-u0"
export GCP_LOCATION="us-central1"
export GCP_ZONE="us-central1-a"
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/llm-service-account.json"

# Build the Docker image
echo "🔨 Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container interactively
echo "🚀 Running container: $IMAGE_NAME"
docker run --rm --name $IMAGE_NAME -ti \
  -v "$BASE_DIR":/app \
  -v "$SECRETS_DIR":/secrets \
  -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
  -e GCP_PROJECT=$GCP_PROJECT \
  -e GCP_LOCATION=$GCP_LOCATION \
  -e GCP_ZONE=$GCP_ZONE \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  $IMAGE_NAME
