"""
model.py
Container-oriented execution layer for the DnD MLOps workflow.

This file exposes 4 high-level functions:
    - run_collector()
    - run_processor()
    - run_trainer()
    - run_deployer()

Each function internally triggers a Docker container via subprocess.
"""

import os
import subprocess


# ============================================================
# Environment Variables
# ============================================================

GCP_PROJECT = os.getenv("GCP_PROJECT", "ac215")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "ac215-ml-workflow")
SECRETS_DIR = os.getenv("SECRETS_DIR", os.path.expanduser("~/Desktop/secrets"))
PERSIST_DIR = os.getenv("PERSIST_DIR", os.path.expanduser("~/Desktop/persistent-folder"))

# Docker images
COLLECTOR_IMAGE = "dnd-collector"
PROCESSOR_IMAGE = "dnd-processor"
TRAINER_IMAGE = "dnd-model-training"
DEPLOY_IMAGE = "dnd-model-deploy"


# ============================================================
# Utility: run docker command
# ============================================================

def _run(cmd_list: list):
    print("\n============================================")
    print("Running command:")
    print(" ".join(cmd_list))
    print("============================================\n")

    subprocess.run(cmd_list, check=True)


# ============================================================
# 1️⃣ Collector
# ============================================================

def run_collector(nums: int = 50, query: str = "dnd dungeon dragon magic tavern"):
    """
    Launches dnd-collector container to scrape / search data.
    """

    cmd = [
        "docker", "run", "--rm",
        "-e", f"GCP_PROJECT={GCP_PROJECT}",
        "-e", f"GCS_BUCKET_NAME={GCS_BUCKET_NAME}",
        "-e", "GOOGLE_APPLICATION_CREDENTIALS=/secrets/ml-workflow.json",
        "-v", f"{SECRETS_DIR}:/secrets",
        COLLECTOR_IMAGE,
        "cli.py",
        "--search",
        "--nums", str(nums),
        "--query", query,
        "--bucket", GCS_BUCKET_NAME,
    ]

    _run(cmd)



# ============================================================
# 2️⃣ Processor
# ============================================================

def run_processor():
    """
    Launches dnd-processor container to clean & split data.
    """

    cmd = [
        "docker", "run", "--rm",
        "-e", f"GCP_PROJECT={GCP_PROJECT}",
        "-e", f"GCS_BUCKET_NAME={GCS_BUCKET_NAME}",
        "-e", "GOOGLE_APPLICATION_CREDENTIALS=/secrets/ml-workflow.json",
        "-v", f"{SECRETS_DIR}:/secrets",
        "-v", f"{PERSIST_DIR}:/persistent",
        PROCESSOR_IMAGE
    ]

    _run(cmd)



# ============================================================
# 3️⃣ Gemini Model Training
# ============================================================

def run_trainer(epochs: int = 3):
    """
    Launches dnd-model-training container to start supervised fine-tuning.
    """

    cmd = [
        "docker", "run", "--rm",
        "-e", f"GCP_PROJECT={GCP_PROJECT}",
        "-e", f"GCS_BUCKET_NAME={GCS_BUCKET_NAME}",
        "-e", "GOOGLE_APPLICATION_CREDENTIALS=/secrets/ml-workflow.json",
        "-v", f"{SECRETS_DIR}:/secrets",
        "-v", f"{PERSIST_DIR}:/persistent",
        TRAINER_IMAGE,
        "tune",
        "--epochs", str(epochs),
    ]

    _run(cmd)



# ============================================================
# 4️⃣ Model Deployment
# ============================================================

def run_deployer():
    """
    Launches deployment container.
    This can use tuned model stored in /persistent/tuning_job.json.
    """

    cmd = [
        "docker", "run", "--rm",
        "-e", f"GCP_PROJECT={GCP_PROJECT}",
        "-e", f"GCS_BUCKET_NAME={GCS_BUCKET_NAME}",
        "-e", "GOOGLE_APPLICATION_CREDENTIALS=/secrets/ml-workflow.json",
        "-v", f"{SECRETS_DIR}:/secrets",
        "-v", f"{PERSIST_DIR}:/persistent",
        DEPLOY_IMAGE,
        "deploy"
    ]

    _run(cmd)



# ============================================================
# End of module
# ============================================================
print("model.py successfully loaded (container-oriented workflow).")
