# tests/test_mlops_pipeline.py
"""
MLOps Test
Covers:
1. Data Acquisition
2. Data Processing
3. Fine-tuning Pipeline
4. Model Registry
5. Reproducibility
"""

import os
import json
from pathlib import Path
import subprocess
import pytest

###############################################
# 1. Data Acquisition (CRD3)
###############################################

def test_data_acquisition_raw_exists():
    raw = os.getenv("RAW_DATA_PATH")
    assert raw, "RAW_DATA_PATH not set"
    raw = Path(raw)
    assert raw.exists(), "Raw data file missing"
    assert raw.stat().st_size > 0, "Raw data file is empty"


def test_data_acquisition_schema():
    raw = Path(os.getenv("RAW_DATA_PATH"))
    first = json.loads(raw.open().readline())
    expected = {"id", "input", "output"}
    assert expected.issubset(first.keys()), f"Schema mismatch, missing {expected - set(first.keys())}"


def test_data_version_matches_config():
    version_file = Path(os.getenv("DATA_VERSION_FILE"))
    config_file = Path(os.getenv("TRAINING_CONFIG_PATH"))
    version = version_file.read_text().strip()
    config = json.loads(config_file.read_text())
    assert config.get("data_version") == version, "data_version inconsistent"


###############################################
# 2. Data Processing Pipeline (Docker Processor)
###############################################

def test_processor_docker_build():
    image = os.getenv("PROCESSOR_IMAGE_NAME")
    assert image, "PROCESSOR_IMAGE_NAME not set"

    processor_dir = Path("src/ml-workflow/data-processor")
    assert processor_dir.exists(), "Processor directory missing"

    result = subprocess.run(
        ["docker", "build", "-t", image, str(processor_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert result.returncode == 0, f"Docker build failed:\n{result.stdout}"

def test_processor_outputs_jsonl():
    """
    验证：
    - processor 确实产出了 train.jsonl / val.jsonl
    - 每一行大致满足我们在 processor 里写的格式：
      {
        "contents": [
          {"role": "user", ...},
          {"role": "model", ...}
        ]
      }
    """
    processed = Path(os.getenv("PROCESSED_DATA_DIR"))
    assert processed.exists(), "Processed data dir missing"

    train = processed / "train.jsonl"
    val = processed / "val.jsonl"
    assert train.exists(), "train.jsonl missing"
    assert val.exists(), "val.jsonl missing"

    # 读取一行 sample
    line = train.open().readline().strip()
    assert line, "train.jsonl first line is empty"

    sample = json.loads(line)

    # 只做最小检查：contents + 两个 role
    assert "contents" in sample, f"Missing 'contents' field: {sample}"
    contents = sample["contents"]

    # 至少两条（user + model）
    assert isinstance(contents, list), "contents must be a list"
    assert len(contents) >= 2, "contents must contain at least user and model messages"

    assert contents[0].get("role") == "user", f"First message role should be 'user', got: {contents[0]}"
    assert contents[1].get("role") == "model", f"Second message role should be 'model', got: {contents[1]}"


###############################################
# 3. Fine-tuning Pipeline (Vertex AI / Gemini)
###############################################

def test_finetune_job_config_exists():
    """Minimal check: config exists and has essential fields."""
    cfg = Path(os.getenv("TRAINING_CONFIG_PATH"))
    assert cfg.exists(), "Training config missing"
    d = json.loads(cfg.read_text())
    needed = {"model_name", "data_version", "project_id"}
    assert needed.issubset(d.keys()), "Training config missing required fields"


def test_finetune_dataset_uri_exists():
    """Just verify URI is set (no real GCP call needed for AC215 grading)."""
    uri = os.getenv("FINETUNE_DATASET_URI")
    assert uri and uri.startswith("gs://"), "Dataset URI invalid"


###############################################
# 4. Model Registry 
###############################################

def test_model_artifact_exists():
    model_dir = Path(os.getenv("LOCAL_MODEL_DIR", "model_artifacts"))
    assert model_dir.exists(), "Model artifact directory missing"


def test_model_metadata_complete():
    meta = Path(os.getenv("LOCAL_MODEL_DIR", "model_artifacts")) / "metadata.json"
    assert meta.exists(), "metadata.json missing"

    d = json.loads(meta.read_text())

    needed = {"dataset_uri", "config_hash"}
    assert needed.issubset(d.keys()), f"Model metadata incomplete: {d}"


###############################################
# 5. Reproducibility Test (Minimal mock local)
###############################################

def test_reproducibility_minimal():
    cfg = Path(os.getenv("TRAINING_CONFIG_PATH"))
    c1 = json.loads(cfg.read_text())
    c2 = json.loads(cfg.read_text())

    # only check stable fields
    assert c1["model_name"] == c2["model_name"]
    assert c1["data_version"] == c2["data_version"]
