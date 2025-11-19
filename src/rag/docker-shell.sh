#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Set vairables
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../persistent-folder/
export SECRETS_DIR=$(pwd)/../../secrets/
export GCP_PROJECT="aic215" # CHANGE TO YOUR PROJECT ID
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/dnd-master.json"
export IMAGE_NAME="dnd-rag-cli"


# Create the network if we don't have it yet
docker network inspect dnd-master-network >/dev/null 2>&1 || docker network create dnd-master-network

# Build the image based on the Dockerfile
#docker build --no-cache -t $IMAGE_NAME -f Dockerfile .
docker build -t $IMAGE_NAME -f Dockerfile .

# Run All Containers
docker-compose run --rm --service-ports $IMAGE_NAME