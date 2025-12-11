import os
import subprocess
from pathlib import Path

# Repo root inside container (because you run with -v /repo)
REPO = Path("/repo")

# DV directory
DV_DIR = REPO / "src/data-versioning"

# The data folder being tracked by DVC
DATA_DIR = DV_DIR / "data"

# raw / processed / splits
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "training_splits"

# test file
TEST_FILE = RAW_DIR / "test_raw.csv"


def run(cmd, cwd=None):
    """Run a shell command and assert success."""
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    assert result.returncode == 0, f"Command failed: {cmd}\n{result.stderr}"
    return result


def test_01_raw_folder_exists():
    assert RAW_DIR.exists(), "raw folder does not exist"
    assert PROCESSED_DIR.exists(), "processed folder does not exist"

def test_02_dvc_add_data_folder():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TEST_FILE.write_text("id,value\n1,100\n")

    # Add entire data folder
    run("dvc add data", cwd=DV_DIR)

    dvc_file = DV_DIR / "data.dvc"
    assert dvc_file.exists(), "data.dvc not created"


def test_03_dvc_push_and_pull():
    run("dvc push", cwd=DV_DIR)

    # delete local test file
    os.remove(TEST_FILE)
    assert not TEST_FILE.exists()

    # restore from remote
    run("dvc pull", cwd=DV_DIR)
    assert TEST_FILE.exists(), "dvc pull did not restore raw file"


def test_04_hash_changes_when_raw_changes():
    """Modifying raw should trigger DVC status reporting change."""

    # Modify raw file
    TEST_FILE.write_text("changed,content\n")

    # Check DVC status detects change
    result = subprocess.run(
        "dvc status",
        shell=True,
        cwd=DV_DIR,
        capture_output=True,
        text=True
    )

    output = result.stdout.lower()
    assert "changed" in output or "modified" in output, \
        "DVC did not detect change in raw data"
