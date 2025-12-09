import os
import argparse
import pandas as pd
import json
import time
import glob
from sklearn.model_selection import train_test_split
from google.cloud import storage
from datasets import load_dataset
import tiktoken

# Gen AI
from google import genai
from google.genai import types
from google.genai.types import Content, Part, GenerationConfig, ToolConfig
from google.genai import errors

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
GENERATIVE_MODEL = "gemini-2.0-flash-001"
OUTPUT_FOLDER = "data"
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

#############################################################################
#                       Initialize the LLM Client                           #
llm_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
#############################################################################

safety_settings = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
]

# System Prompt for D&D Narrator/NPC
SYSTEM_INSTRUCTION = """You are an immersive Dungeons & Dragons narrator and NPC character roleplayer. Your role is to bring the world of D&D to life through vivid storytelling, compelling character voices, and engaging narrative descriptions. You will respond to player actions and questions in character, creating memorable encounters and advancing the story.

Guidelines:
1. Narrative Style:
   - Use rich, descriptive language to paint vivid scenes
   - Engage all five senses in your descriptions
   - Create atmosphere and tension appropriate to the situation
   - Balance narration with character dialogue

2. Character Roleplaying:
   - Embody diverse NPC personalities (merchants, guards, villains, allies, etc.)
   - Give each character a distinct voice and mannerisms
   - Stay in character consistently
   - React authentically to player actions and decisions

3. Tone and Atmosphere:
   - Adapt tone to match the scene (mysterious, comedic, tense, epic, etc.)
   - Build suspense before important moments
   - Use dramatic pauses and emphasis effectively
   - Create emotional resonance with storytelling

4. D&D Knowledge:
   - Reference D&D lore, rules, and mechanics naturally
   - Describe spell effects, combat actions, and skill checks cinematically
   - Mention monsters, locations, and items from D&D canon
   - Stay true to fantasy genre conventions

5. Player Engagement:
   - Respond to player questions and actions
   - Provide hooks for further exploration
   - Present meaningful choices
   - Keep the story moving forward

This is for fine-tuning purposes based on actual D&D gameplay transcripts."""


def count_tokens(text):
    """
    Count tokens in text using tiktoken (GPT-4 tokenizer as approximation for Gemini)

    Args:
        text: String to count tokens in

    Returns:
        int: Number of tokens
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback: rough estimate of ~4 characters per token
        return len(text) // 4


def load_crd3():
    """Load CRD3 dataset from Hugging Face"""
    print("Loading CRD3 dataset from Hugging Face...")

    # Make dataset folders
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load the CRD3 dataset from Microsoft
    # Using trust_remote_code=True since it has custom loading scripts
    dataset = load_dataset("microsoft/crd3", trust_remote_code=True)

    print(f"Dataset loaded successfully!")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")

    return dataset


def prepare(max_samples=1000, use_all=False):
    """
    Prepare CRD3 dataset for Gemini fine-tuning

    Args:
        max_samples: Maximum number of samples to use from the dataset (ignored if use_all=True)
        use_all: If True, process all train and test data from the dataset
    """
    print("prepare()")

    # Load the CRD3 dataset
    dataset = load_crd3()

    # Process the dataset
    # CRD3 has fields: chunk, chunk_id, turn_num, turn_start, turn_end, names, utterances, etc.

    if use_all:
        print("Processing ALL data from CRD3 dataset (all splits)...")
        print(f"Available splits: {list(dataset.keys())}")

        train_processed = []
        validation_processed = []
        test_processed = []

        # Process train split
        if "train" in dataset:
            print(f"Processing train split ({len(dataset['train'])} samples)...")
            for i, sample in enumerate(dataset["train"]):
                chunk = sample["chunk"]
                if chunk and len(chunk.strip()) > 50:
                    context = f"Continue the D&D story and narrate what happens next:"
                    train_processed.append({"context": context, "response": chunk})
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{len(dataset['train'])} train samples")

        # Process validation split (this will be used as Gemini's validation data)
        if "validation" in dataset:
            print(f"Processing validation split ({len(dataset['validation'])} samples)...")
            for i, sample in enumerate(dataset["validation"]):
                chunk = sample["chunk"]
                if chunk and len(chunk.strip()) > 50:
                    context = f"Continue the D&D story and narrate what happens next:"
                    validation_processed.append({"context": context, "response": chunk})
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{len(dataset['validation'])} validation samples")

        # Process test split (for future evaluation, not used in fine-tuning)
        if "test" in dataset:
            print(f"Processing test split ({len(dataset['test'])} samples)...")
            for i, sample in enumerate(dataset["test"]):
                chunk = sample["chunk"]
                if chunk and len(chunk.strip()) > 50:
                    context = f"Continue the D&D story and narrate what happens next:"
                    test_processed.append({"context": context, "response": chunk})
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{len(dataset['test'])} test samples")

        print(f"Total train samples: {len(train_processed)}")
        print(f"Total validation samples: {len(validation_processed)}")
        print(f"Total test samples: {len(test_processed)}")

        # Convert to DataFrames
        df_train = pd.DataFrame(train_processed) if train_processed else pd.DataFrame()
        df_validation = pd.DataFrame(validation_processed) if validation_processed else pd.DataFrame()
        df_test = pd.DataFrame(test_processed) if test_processed else pd.DataFrame()

        # Clean
        df_train = df_train.dropna()
        df_validation = df_validation.dropna()
        df_test = df_test.dropna()

        print(f"After cleaning - Train: {len(df_train)}, Validation: {len(df_validation)}, Test: {len(df_test)}")

        # Combine for full dataset CSV
        df_all = pd.concat([df_train, df_validation, df_test], ignore_index=True)

    else:
        print(f"Processing CRD3 dataset (max {max_samples} samples from train split)...")
        processed_data = []

        # Use the train split
        train_data = dataset["train"]

        # Limit the number of samples to process
        num_samples = min(max_samples, len(train_data))

        for i in range(num_samples):
            sample = train_data[i]
            chunk = sample["chunk"]

            if chunk and len(chunk.strip()) > 50:
                context = f"Continue the D&D story and narrate what happens next:"
                processed_data.append({"context": context, "response": chunk})

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples} samples")

        print(f"Total processed samples: {len(processed_data)}")

        # Convert to DataFrame
        df_all = pd.DataFrame(processed_data)
        df_all = df_all.dropna()

        # Train/test split for limited samples mode
        df_train, df_validation = train_test_split(df_all, test_size=0.1, random_state=42)
        df_test = pd.DataFrame()  # No test set in limited mode

    # Summary
    if use_all:
        print(f"Final - Train: {len(df_train)}, Validation: {len(df_validation)}, Test: {len(df_test)}")
    else:
        print(f"Final - Train: {len(df_train)}, Validation: {len(df_validation)}")
    print("Total dataset shape:", df_all.shape)
    print(df_all.head())

    # Count tokens for cost estimation
    print("\n" + "=" * 60)
    print("TOKEN COUNT ANALYSIS FOR COST ESTIMATION")
    print("=" * 60)

    print("\nCounting tokens (this may take a moment)...")

    # Count tokens in each split
    train_tokens = df_train["response"].apply(count_tokens).sum()
    train_context_tokens = df_train["context"].apply(count_tokens).sum()
    train_total_tokens = train_tokens + train_context_tokens

    validation_tokens = df_validation["response"].apply(count_tokens).sum()
    validation_context_tokens = df_validation["context"].apply(count_tokens).sum()
    validation_total_tokens = validation_tokens + validation_context_tokens

    # Use only 256 validation samples for Gemini
    validation_limited_tokens = df_validation[:256]["response"].apply(count_tokens).sum()
    validation_limited_context_tokens = df_validation[:256]["context"].apply(count_tokens).sum()
    validation_limited_total_tokens = validation_limited_tokens + validation_limited_context_tokens

    # Calculate per-sample averages
    avg_tokens_per_sample = train_total_tokens / len(df_train) if len(df_train) > 0 else 0

    print(f"\n📊 Token Counts:")
    print(f"  Training set:")
    print(f"    - Total tokens: {train_total_tokens:,}")
    print(f"    - Average tokens/sample: {avg_tokens_per_sample:.0f}")
    print(f"    - Response tokens: {train_tokens:,}")
    print(f"    - Context tokens: {train_context_tokens:,}")

    print(f"\n  Validation set (full {len(df_validation)} samples):")
    print(f"    - Total tokens: {validation_total_tokens:,}")

    print(f"\n  Validation set (256 samples used by Gemini):")
    print(f"    - Total tokens: {validation_limited_total_tokens:,}")

    # Calculate cost estimates
    tokens_per_epoch = train_total_tokens + validation_limited_total_tokens

    print(f"\n💰 COST ESTIMATION:")
    print(f"  Tokens per training epoch: {tokens_per_epoch:,}")
    print(f"  Price: $0.008 per 1,000 tokens")
    print(f"\n  Estimated costs by number of epochs:")
    for epochs in [1, 2, 3, 4, 5]:
        total_tokens = tokens_per_epoch * epochs
        cost = (total_tokens / 1000) * 0.008
        print(f"    {epochs} epoch(s):  {total_tokens:,} tokens = ${cost:.2f}")

    print("\n  📌 Recommended: 3-4 epochs for optimal results")
    print("=" * 60 + "\n")

    # Save the full dataset
    filename = os.path.join(OUTPUT_FOLDER, "dnd-instruct-dataset.csv")
    df_all.to_csv(filename, index=False)

    # Build training formats for train, validation, and test
    df_train["text"] = "user: " + df_train["context"] + "\n" + "assistant: " + df_train["response"]
    df_validation["text"] = "user: " + df_validation["context"] + "\n" + "assistant: " + df_validation["response"]
    if len(df_test) > 0:
        df_test["text"] = "user: " + df_test["context"] + "\n" + "assistant: " + df_test["response"]

    # Gemini Data prep: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning-prepare
    df_train["contents"] = df_train.apply(
        lambda row: [
            {"role": "user", "parts": [{"text": row["context"]}]},
            {"role": "model", "parts": [{"text": row["response"]}]},
        ],
        axis=1,
    )
    df_validation["contents"] = df_validation.apply(
        lambda row: [
            {"role": "user", "parts": [{"text": row["context"]}]},
            {"role": "model", "parts": [{"text": row["response"]}]},
        ],
        axis=1,
    )
    if len(df_test) > 0:
        df_test["contents"] = df_test.apply(
            lambda row: [
                {"role": "user", "parts": [{"text": row["context"]}]},
                {"role": "model", "parts": [{"text": row["response"]}]},
            ],
            axis=1,
        )

    # Save CSV files for all three splits
    df_train[["text"]].to_csv(os.path.join(OUTPUT_FOLDER, "train.csv"), index=False)
    df_validation[["text"]].to_csv(os.path.join(OUTPUT_FOLDER, "validation.csv"), index=False)
    if len(df_test) > 0:
        df_test[["text"]].to_csv(os.path.join(OUTPUT_FOLDER, "test.csv"), index=False)

    # Gemini: Max numbers of examples in validation dataset: 256
    df_validation_limited = df_validation[:256]

    # Save JSONL files for Gemini fine-tuning
    # Train: all samples
    with open(os.path.join(OUTPUT_FOLDER, "train.jsonl"), "w") as json_file:
        json_file.write(df_train[["contents"]].to_json(orient="records", lines=True))

    # Validation: limited to 256 for Gemini (but save full version too)
    with open(os.path.join(OUTPUT_FOLDER, "validation.jsonl"), "w") as json_file:
        json_file.write(df_validation_limited[["contents"]].to_json(orient="records", lines=True))

    with open(os.path.join(OUTPUT_FOLDER, "validation_full.jsonl"), "w") as json_file:
        json_file.write(df_validation[["contents"]].to_json(orient="records", lines=True))

    # Test: all samples if exists
    if len(df_test) > 0:
        with open(os.path.join(OUTPUT_FOLDER, "test.jsonl"), "w") as json_file:
            json_file.write(df_test[["contents"]].to_json(orient="records", lines=True))

    print("Dataset preparation complete!")
    print(f"Files saved to {OUTPUT_FOLDER}/")


def upload():
    print("upload()")

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    timeout = 300

    data_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.jsonl")) + glob.glob(os.path.join(OUTPUT_FOLDER, "*.csv"))
    data_files.sort()

    # Upload to a D&D-specific folder
    for index, data_file in enumerate(data_files):
        filename = os.path.basename(data_file)
        destination_blob_name = os.path.join("dnd-narrator-finetune-dataset", filename)
        blob = bucket.blob(destination_blob_name)
        print("Uploading file:", data_file, destination_blob_name)
        blob.upload_from_filename(data_file, timeout=timeout)

    print(f"Uploaded {len(data_files)} files to gs://{GCS_BUCKET_NAME}/dnd-narrator-finetune-dataset/")


def main(args=None):
    print("CLI Arguments:", args)

    if args.prepare:
        prepare(max_samples=args.max_samples, use_all=args.all)

    if args.upload:
        upload()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal '--help', it will provide the description
    parser = argparse.ArgumentParser(
        description="DnD Narrator Dataset Creator - Load CRD3 and prepare for Gemini fine-tuning"
    )

    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Load CRD3 dataset from HuggingFace and prepare it for fine-tuning",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to use from CRD3 dataset (default: 1000, ignored if --all is used)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process ALL data from CRD3 (both train and test splits, ignores --max-samples)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload prepared data to GCS bucket",
    )

    args = parser.parse_args()

    main(args)
