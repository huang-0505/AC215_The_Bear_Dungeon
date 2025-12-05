#!/bin/bash

# Exit on error
set -e

# DnD Master Test Environment
# Builds and enters the test container

# Get the directory of this script (tests/) and go to project root
export BASE_DIR=$(cd $(dirname $0)/.. && pwd)
export IMAGE_NAME="dnd-test-runner"

# Build the image from project root with tests/Dockerfile
echo "Building test container..."
docker build -t $IMAGE_NAME -f "$BASE_DIR/tests/Dockerfile" "$BASE_DIR"

# Run container
echo "Entering test container..."
docker run --rm -it \
    -v "$BASE_DIR/src":/app/src:ro \
    -v "$BASE_DIR/tests":/app/tests:ro \
    -v "$BASE_DIR/conftest.py":/app/conftest.py:ro \
    -v "$BASE_DIR/pytest.ini":/app/pytest.ini:ro \
    -v "$BASE_DIR/htmlcov":/app/htmlcov \
    -v "$BASE_DIR/coverage.xml":/app/coverage.xml \
    --network host \
    $IMAGE_NAME \
    /bin/bash
