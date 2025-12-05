#!/bin/bash

# Exit on error
set -e

# Get the directory of this script (tests/) and go to project root
export BASE_DIR=$(cd $(dirname $0)/.. && pwd)
export IMAGE_NAME="dnd-test-runner"

# Build the image from project root with tests/Dockerfile
echo "Building test container..."
docker build -t $IMAGE_NAME -f "$BASE_DIR/tests/Dockerfile" "$BASE_DIR"

# Run tests based on argument
TEST_TYPE=${1:-all}

case $TEST_TYPE in
    unit)
        echo "Running unit tests..."
        docker run --rm \
            -v "$BASE_DIR/htmlcov":/app/htmlcov \
            -v "$BASE_DIR/coverage.xml":/app/coverage.xml \
            $IMAGE_NAME \
            uv run --directory /app/tests pytest /app/tests/unit -v -m unit --tb=short
        ;;

    integration)
        echo "Running integration tests..."
        docker run --rm \
            -v "$BASE_DIR/htmlcov":/app/htmlcov \
            -v "$BASE_DIR/coverage.xml":/app/coverage.xml \
            $IMAGE_NAME \
            uv run --directory /app/tests pytest /app/tests/integration -v -m integration --tb=short
        ;;

    system)
        echo "System tests require running services. Use ../run-tests.sh system instead."
        exit 1
        ;;

    coverage)
        echo "Generating coverage report..."
        docker run --rm \
            -v "$BASE_DIR/htmlcov":/app/htmlcov \
            -v "$BASE_DIR/coverage.xml":/app/coverage.xml \
            $IMAGE_NAME \
            uv run --directory /app/tests pytest /app/tests/unit /app/tests/integration \
                --cov=/app/src/backend-tmp/api \
                --cov=/app/src/rule_agent \
                --cov=/app/src/orchestrator \
                --cov-report=html \
                --cov-report=xml \
                --cov-report=term
        echo "Coverage report generated in htmlcov/"
        ;;

    all)
        echo "Running all unit and integration tests..."
        docker run --rm \
            -v "$BASE_DIR/htmlcov":/app/htmlcov \
            -v "$BASE_DIR/coverage.xml":/app/coverage.xml \
            $IMAGE_NAME \
            uv run --directory /app/tests pytest /app/tests/unit /app/tests/integration -v --tb=short
        ;;

    *)
        echo "Usage: $0 [unit|integration|coverage|all]"
        echo ""
        echo "Options:"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests only"
        echo "  coverage    - Run tests with coverage report"
        echo "  all         - Run all unit and integration tests (default)"
        echo ""
        echo "Note: For system tests, use ../run-tests.sh system"
        exit 1
        ;;
esac
