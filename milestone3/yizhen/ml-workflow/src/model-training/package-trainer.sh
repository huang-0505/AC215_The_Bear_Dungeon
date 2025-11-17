#!/bin/bash
set -e

# ======== Basic Configuration ========
export GCP_PROJECT="even-turbine-471117-u0"
export GCS_PACKAGE_URI="gs://even-turbine-471117-u0/model-training"
export PACKAGE_NAME="dnd-model-trainer"
export PACKAGE_DIR="trainer"     # 你的 Python 包目录
export OUTPUT_TAR="${PACKAGE_NAME}.tar.gz"

echo "============================================"
echo "📦 Packaging Python Trainer for Vertex AI"
echo "Project: $GCP_PROJECT"
echo "Package Directory: $PACKAGE_DIR"
echo "Target URI: $GCS_PACKAGE_URI/$OUTPUT_TAR"
echo "============================================"

# ======== Step 1: Clean Old Files ========
echo "🧹 Cleaning old artifacts..."
rm -f trainer.tar trainer.tar.gz $OUTPUT_TAR

# ======== Step 2: Create TAR.GZ ========
echo "📦 Creating tar.gz package..."
# tar MUST contain trainer/ at root, not trainer/trainer
tar -czvf $OUTPUT_TAR $PACKAGE_DIR

# ======== Step 3: Upload to GCS ========
echo "🚀 Uploading to GCS..."
gsutil cp $OUTPUT_TAR $GCS_PACKAGE_URI/$OUTPUT_TAR

echo "============================================"
echo "✅ Trainer package upload complete!"
echo "Uploaded to: $GCS_PACKAGE_URI/$OUTPUT_TAR"
echo "============================================"
