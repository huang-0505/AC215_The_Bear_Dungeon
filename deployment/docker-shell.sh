#!/bin/bash

# exit immediately if a command exits with a non-zero status
#set -e

# Define some environment variables
export IMAGE_NAME="dnd-master-deployment"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../secrets/
export SSH_DIR=$(pwd)/../../secrets/
# export SECRETS_DIR=$(pwd)/../../secrets/ac215-project/
# export SSH_DIR=$(pwd)/../../secrets/ac215-project/
export GCP_PROJECT="even-turbine-471117-u0" # Change to your GCP Project
export GCP_REGION="us-central1"
export GCP_ZONE="us-central1-a"
export GOOGLE_APPLICATION_CREDENTIALS=/secrets/deployment.json
export PULUMI_BUCKET="gs://$GCP_PROJECT-pulumi-state-bucket"

# Create local Pulumi plugins directory if it doesn't exist
mkdir -p $BASE_DIR/pulumi-plugins

# Check if container is already running
if docker ps --format "table {{.Names}}" | grep -q "^${IMAGE_NAME}$"; then
    echo "Container '${IMAGE_NAME}' is already running. Shelling into existing container..."
    docker exec -it $IMAGE_NAME /bin/bash ./docker-entrypoint.sh
else
    echo "Container '${IMAGE_NAME}' is not running. Building and starting new container..."

    # Build the image based on the Dockerfile
    #docker build -t $IMAGE_NAME -f Dockerfile .
    docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .

    # Run the container
    # Mount project root so we can access src/ directory
    # From deployment, go up 1 level to reach project root
    PROJECT_ROOT=$(cd "$BASE_DIR/.." && pwd)
    docker run --rm --name $IMAGE_NAME -ti \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$BASE_DIR":/app \
    -v "$PROJECT_ROOT":/project-root \
    -v "$SECRETS_DIR":/secrets \
    -v "$SSH_DIR/.ssh":/home/app/.ssh:ro \
    -v "$(pwd)/docker_config.json":/root/.docker/config.json \
    -v "$(pwd)/pulumi-plugins":/root/.pulumi/plugins \
    -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
    -e USE_GKE_GCLOUD_AUTH_PLUGIN=True \
    -e GCP_PROJECT=$GCP_PROJECT \
    -e GCP_REGION=$GCP_REGION \
    -e GCP_ZONE=$GCP_ZONE \
    -e PULUMI_BUCKET=$PULUMI_BUCKET \
    $IMAGE_NAME
fi
