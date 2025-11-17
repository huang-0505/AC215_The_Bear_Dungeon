import os
import json
from typing import Tuple, List
from google.cloud import storage

# ========== Configuration ==========
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "ac215")
TARGET_BUCKET = os.getenv("GCS_BUCKET_NAME", "ac215-ml-workflow")

# 同学的源文件（从这个 bucket 复制）
SOURCE_URIS = [
    "gs://dnd-master-dataset/dnd-narrator-finetune-dataset/train.jsonl",
    "gs://dnd-master-dataset/dnd-narrator-finetune-dataset/validation.jsonl",
]

# 你希望复制到自己的路径
TARGET_PREFIX = "dnd-ml-dataset"


# ========== Utility functions ==========
def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    """Parse a GCS URI into (bucket, blob_path)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""
    return bucket, blob_path


# ========== Copy Logic ==========
def copy_one_file(client: storage.Client, source_uri: str, target_bucket: str, target_prefix: str) -> str:
    """Copy a single file between buckets."""
    src_bucket_name, src_blob_name = parse_gcs_uri(source_uri)
    src_bucket = client.bucket(src_bucket_name)
    src_blob = src_bucket.blob(src_blob_name)

    filename = os.path.basename(src_blob_name)
    dest_blob_name = f"{target_prefix}/{filename}"

    dest_bucket = client.bucket(target_bucket)
    dest_blob = dest_bucket.blob(dest_blob_name)

    print(f"📥 Copying {source_uri} → gs://{target_bucket}/{dest_blob_name}")
    dest_blob.rewrite(src_blob)

    return f"gs://{target_bucket}/{dest_blob_name}"


def copy_all_files() -> List[str]:
    """Copy all files from SOURCE_URIS to your bucket."""
    print("🚀 Starting dataset copy...")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Target bucket: {TARGET_BUCKET}")
    print(f"Credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")

    client = storage.Client(project=PROJECT_ID)
    dest_uris = []

    for uri in SOURCE_URIS:
        dest_uris.append(copy_one_file(client, uri, TARGET_BUCKET, TARGET_PREFIX))

    print("\n✅ All files copied successfully:")
    for d in dest_uris:
        print(f"   → {d}")

    return dest_uris


# ========== Verify JSONL ==========
def verify_jsonl_file(client: storage.Client, gcs_uri: str, max_lines: int = 3):
    """Verify that a GCS JSONL file contains valid JSON lines."""
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    blob = client.bucket(bucket_name).blob(blob_name)

    print(f"\n🔍 Verifying JSONL file: {gcs_uri}")
    content = blob.download_as_text()
    lines = content.strip().split("\n")

    if not lines:
        print("⚠️ File is empty.")
        return

    for i, line in enumerate(lines[:max_lines]):
        try:
            data = json.loads(line)
            print(f"   ✅ Line {i+1}: valid JSON with keys {list(data.keys())}")
        except json.JSONDecodeError as e:
            print(f"   ❌ Line {i+1} invalid JSON: {e}")
            return

    print("✅ JSONL structure looks valid.")


def verify_copied_files(dest_uris: List[str]):
    """Verify all copied JSONL files."""
    client = storage.Client(project=PROJECT_ID)
    for uri in dest_uris:
        if uri.endswith(".jsonl"):
            verify_jsonl_file(client, uri)


# ========== Entry Point ==========
if __name__ == "__main__":
    print("📦 Running DnD dataset downloader...")

    try:
        copied = copy_all_files()
        verify_copied_files(copied)
        print("\n🎯 Data copy + verification complete!")
    except Exception as e:
        print(f"❌ Error: {e}")
