import os
import json
import argparse
from google.cloud import storage
from datasets import load_dataset

# Environment variables from pipeline
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

RAW_PREFIX = "data/collector/"  # where we store raw CRD3


def load_crd3():
    print("📥 Loading CRD3 dataset from HuggingFace (raw)...")
    dataset = load_dataset("microsoft/crd3", trust_remote_code=True)
    print("✅ CRD3 dataset loaded.")
    return dataset


def write_split_to_local(name, data, local_path):
    """
    Writes raw HF split directly into JSONL without modification.
    """
    print(f"📝 Writing {name} split → {local_path}")

    with open(local_path, "w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")

    print(f"   ✔ Finished {name}")


def upload_file_to_gcs(local_path, bucket_name, blob_path):
    """
    Upload using blob.upload_from_filename (consistent with your existing code).
    """
    print(f"⬆️  Uploading {local_path} → gs://{bucket_name}/{blob_path}")

    storage_client = storage.Client(project=GCP_PROJECT)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

    print("   ✔ Upload complete")


def run_collector():
    dataset = load_crd3()

    # Local temp paths in the container
    local_train = "/app/raw_crd3_train.jsonl"
    local_valid = "/app/raw_crd3_validation.jsonl"
    local_test = "/app/raw_crd3_test.jsonl"

    # Write raw JSONL (no cleaning)
    write_split_to_local("train", dataset["train"], local_train)

    if "validation" in dataset:
        write_split_to_local("validation", dataset["validation"], local_valid)
    else:
        print("⚠️ No validation split found in CRD3")

    write_split_to_local("test", dataset["test"], local_test)

    # Upload to GCS
    upload_file_to_gcs(local_train, GCS_BUCKET_NAME, RAW_PREFIX + "raw_crd3_train.jsonl")

    if "validation" in dataset:
        upload_file_to_gcs(local_valid, GCS_BUCKET_NAME, RAW_PREFIX + "raw_crd3_validation.jsonl")

    upload_file_to_gcs(local_test, GCS_BUCKET_NAME, RAW_PREFIX + "raw_crd3_test.jsonl")

    print("\n🎉 Collector completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRD3 Raw Data Collector Component")
    parser.add_argument("--run", action="store_true", help="Run collector end-to-end")

    args = parser.parse_args()
    if args.run:
        run_collector()
    else:
        print("Use --run to execute collector")
