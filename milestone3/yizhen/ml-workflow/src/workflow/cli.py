"""
CLI launcher for DnD Narrator MLOps workflow.
This orchestrates 4 containers:
1. dnd-collector
2. dnd-processor
3. dnd-model-training
4. dnd-model-deploy
"""

import argparse
import subprocess
import sys
import os


# ============================================================
# Global config
# ============================================================

GCP_PROJECT = os.getenv("GCP_PROJECT", "ac215")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "ac215-ml-workflow")

COMMON_ENV = [
    "-e", f"GCP_PROJECT={GCP_PROJECT}",
    "-e", f"GCS_BUCKET_NAME={GCS_BUCKET_NAME}",
    "-e", "GOOGLE_APPLICATION_CREDENTIALS=/secrets/ml-workflow.json",
    "-v", "/secrets:/secrets",
    "-v", "/persistent:/persistent",
]

COLLECTOR = "dnd-collector"
PROCESSOR = "dnd-processor"
TRAINER = "dnd-model-training"
DEPLOY = "dnd-model-deploy"


# ============================================================
# Utilities
# ============================================================

def run(cmd: list):
    """Wrapper for docker calls"""
    print(f"\n=== RUNNING: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Step failed.")
        sys.exit(1)


# ============================================================
# Single-step runners
# ============================================================

def run_collector(_):
    run(["docker", "run", "--rm"] + COMMON_ENV + [COLLECTOR])


def run_processor(_):
    run(["docker", "run", "--rm"] + COMMON_ENV + [PROCESSOR])


def run_trainer(_):
    # default 3 epochs
    run(["docker", "run", "--rm"] + COMMON_ENV + [
        TRAINER,
        "tune",
        "--epochs", "3",
    ])


def run_deploy(_):
    run(["docker", "run", "--rm"] + COMMON_ENV + [
        DEPLOY,
        "deploy",
    ])


# ============================================================
# Full pipeline runner
# ============================================================

def run_all(_):
    print("\n============================================")
    print("🚀 STARTING FULL DnD MLOPS PIPELINE")
    print("============================================")

    run_collector(None)
    run_processor(None)
    run_trainer(None)
    run_deploy(None)

    print("\n============================================")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("============================================")


# ============================================================
# CLI Entrypoint
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DnD Narrator MLOps Container-Orchestration CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collector", help="Run data collector").set_defaults(func=run_collector)
    subparsers.add_parser("processor", help="Run data processor").set_defaults(func=run_processor)
    subparsers.add_parser("trainer", help="Run model trainer").set_defaults(func=run_trainer)
    subparsers.add_parser("deploy", help="Run model deployer").set_defaults(func=run_deploy)
    subparsers.add_parser("run-all", help="Run full pipeline").set_defaults(func=run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
