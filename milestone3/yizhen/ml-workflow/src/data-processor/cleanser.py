import os
import json
import random
from typing import List, Dict, Tuple
from google.cloud import storage


# ============================
# Environment / Global config
# ============================
GCP_PROJECT = os.getenv("GCP_PROJECT", "ac215")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "ac215-ml-workflow")

RAW_PREFIX = "dnd-raw"
PROCESSED_PREFIX = "dnd-processed"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "99999999"))


# ============================
# Helpers
# ============================

def _client() -> storage.Client:
    return storage.Client(project=GCP_PROJECT)


def _read_jsonl(bucket, blob_name: str) -> List[Dict]:
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"❌ File not found: gs://{bucket.name}/{blob_name}")

    print(f"📥 Reading RAW → gs://{bucket.name}/{blob_name}")

    text = blob.download_as_text()
    lines = text.strip().split("\n")

    data = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            data.append(json.loads(line))
        except Exception as e:
            print(f"⚠️ JSON decode error on line {i}: {e}")

    print(f"📦 Loaded {len(data)} raw samples\n")
    return data


def _write_jsonl(bucket, blob_name: str, data: List[Dict]):
    print(f"📤 Writing {len(data)} → gs://{bucket.name}/{blob_name}")

    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in data),
        content_type="application/jsonl",
    )

    print("✅ Upload complete.\n")


# ============================
# Cleaning
# ============================

def clean_raw_records(records: List[Dict]) -> List[Dict]:
    cleaned = []

    for r in records:
        contents = r.get("contents")
        if not isinstance(contents, list) or len(contents) < 2:
            continue

        try:
            user_msg = contents[0]["parts"][0]["text"].strip()
            model_msg = contents[1]["parts"][0]["text"].strip()

            if user_msg and model_msg:
                cleaned.append(
                    {
                        "contents": [
                            {"role": "user", "parts": [{"text": user_msg}]},
                            {"role": "model", "parts": [{"text": model_msg}]},
                        ]
                    }
                )
        except Exception as e:
            print(f"⚠️ Cleaning error: {e}")
            continue

    print(f"🧹 Cleaning result: {len(cleaned)}/{len(records)} kept.\n")
    return cleaned


# ============================
# Split
# ============================

def split_records(records: List[Dict], n: int) -> Tuple[List[Dict], List[Dict]]:
    random.seed(42)

    if len(records) < n:
        n = len(records)

    subset = records[:n]
    random.shuffle(subset)

    split_at = int(0.9 * n)
    train = subset[:split_at]
    validation = subset[split_at:]

    print(f"🔀 Split → {len(train)} train | {len(validation)} validation\n")
    return train, validation


# ============================
# MAIN PROCESSOR
# ============================

def run_cleaning():
    print("🚀 Running Data Processor...")
    print(f"Bucket = gs://{GCS_BUCKET_NAME}")
    print(f"CHUNK_SIZE = {CHUNK_SIZE}")

    client = _client()
    bucket = client.bucket(GCS_BUCKET_NAME)

    raw_blob = f"{RAW_PREFIX}/raw.jsonl"
    raw_records = _read_jsonl(bucket, raw_blob)

    cleaned_records = clean_raw_records(raw_records)
    train, validation = split_records(cleaned_records, CHUNK_SIZE)

    _write_jsonl(bucket, f"{PROCESSED_PREFIX}/train.jsonl", train)
    _write_jsonl(bucket, f"{PROCESSED_PREFIX}/validation.jsonl", validation)

    print("🎉 Cleaning complete!")
    print(f"Processed data → gs://{GCS_BUCKET_NAME}/{PROCESSED_PREFIX}/")


if __name__ == "__main__":
    run_cleaning()
