import os
import json
import time
from google import genai
from google.genai import types

# ==============================
# Environment Variables
# ==============================
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

TRAIN_URI = f"gs://{GCS_BUCKET_NAME}/dnd-processed/train.jsonl"
VAL_URI   = f"gs://{GCS_BUCKET_NAME}/dnd-processed/validation.jsonl"

BASE_MODEL = "models/gemini-2.0-flash-001"

# Persistent folder for job tracking
PERSIST_DIR = "/persistent"
os.makedirs(PERSIST_DIR, exist_ok=True)

JOB_INFO_FILE = os.path.join(PERSIST_DIR, "tuning_job.json")

# Initialize Gemini client
llm_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)


# ============================================================
# Create tuning job
# ============================================================
def start_tuning(epochs=3):
    print("=== Starting Gemini Supervised Fine-Tuning ===")
    print(f"Train URI: {TRAIN_URI}")
    print(f"Validation URI: {VAL_URI}")
    print(f"Epochs: {epochs}")

    training_dataset   = types.TuningDataset(gcs_uri=TRAIN_URI)
    validation_dataset = types.TuningDataset(gcs_uri=VAL_URI)

    config = types.CreateTuningJobConfig(
        epoch_count = epochs,
        learning_rate_multiplier = 1.0,
        tuned_model_display_name = "dnd-gemini-narrator-v1",
        adapter_size = "ADAPTER_SIZE_FOUR",
        validation_dataset = validation_dataset,
    )

    job = llm_client.tunings.tune(
        base_model = BASE_MODEL,
        training_dataset = training_dataset,
        config = config,
    )

    # Save metadata
    job_info = {
        "job_name": job.name,
        "train_uri": TRAIN_URI,
        "val_uri": VAL_URI,
        "epochs": epochs,
        "status": job.state,
    }
    with open(JOB_INFO_FILE, "w") as f:
        json.dump(job_info, f, indent=2)

    print(f"✓ Job started: {job.name}")
    return job.name


# ============================================================
# Check job status
# ============================================================
def check_status(job_name=None):
    if not os.path.exists(JOB_INFO_FILE):
        raise RuntimeError("No tuning_job.json found in /persistent")

    with open(JOB_INFO_FILE) as f:
        info = json.load(f)

    job_name = job_name or info["job_name"]

    job = llm_client.tunings.get(name=job_name)

    print(f"=== Tuning Job Status ===")
    print(f"Job:    {job_name}")
    print(f"State:  {job.state}")

    # Update info
    info["status"] = job.state

    if job.tuned_model:
        info["tuned_model"] = job.tuned_model.model

    with open(JOB_INFO_FILE, "w") as f:
        json.dump(info, f, indent=2)

    return job.state


# ============================================================
# Get tuned model name
# ============================================================
def get_tuned_model_name():
    if not os.path.exists(JOB_INFO_FILE):
        raise RuntimeError("No tuning job found.")

    with open(JOB_INFO_FILE) as f:
        info = json.load(f)

    if "tuned_model" not in info:
        raise RuntimeError("Model not finished fine-tuning yet.")

    return info["tuned_model"]


# ============================================================
# Wait until completion
# ============================================================
def wait_until_complete(job_name):
    print("=== Waiting for tuning job to complete ===")

    final_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
    }

    while True:
        state = check_status(job_name)
        if state in final_states:
            print(f"✓ Final state: {state}")
            return state
        time.sleep(60)


# ============================================================
# Chat with fine-tuned model
# ============================================================
def call_model_chat(model_name, text_prompt):
    response = llm_client.models.generate_content(
        model=model_name,
        contents=text_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=1500,
            temperature=0.7,
            top_p=0.95,
        ),
    )
    return response.text
