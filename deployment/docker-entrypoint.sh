#!/bin/bash

echo "Container is running!!!"
echo "Architecture: $(uname -m)"
echo "Environment ready! Virtual environment activated."
echo "Python version: $(python --version)"
echo "UV version: $(uv --version)"

# Authenticate gcloud using service account
gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
gcloud config set project $GCP_PROJECT
# login to artifact-registry (both regions)
gcloud auth configure-docker us-docker.pkg.dev --quiet
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
# Check if the bucket exists
if ! gsutil ls -b $PULUMI_BUCKET >/dev/null 2>&1; then
    echo "Bucket does not exist. Creating..."
    gsutil mb -p $GCP_PROJECT $PULUMI_BUCKET
else
    echo "Bucket already exists. Skipping creation."
fi

echo "Logging into Pulumi using GCS bucket: $PULUMI_BUCKET"
pulumi login $PULUMI_BUCKET

# List available stacks
echo "Available Pulumi stacks in GCS:"
gsutil ls $PULUMI_BUCKET/.pulumi/stacks/  || echo "No stacks found."

# Run Bash for interactive mode
/bin/bash