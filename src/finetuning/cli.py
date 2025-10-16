import os
import argparse
from typing import Dict, Any

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer
)
from peft import LoraConfig, PeftModel, TaskType
from trl import SFTTrainer, TrainingArguments


# ------------- Helper: build a unified text sample from various schemas -------------
def build_text_from_example(ex: Dict[str, Any], text_field: str) -> str:
    """
    Convert a dataset example into a single training text string.

    Supported formats:
      1) Single field text: example[text_field] (default: "text")
      2) Instruction-tuning: instruction / input / output -> formatted prompt-response

    Returns:
      A single string that SFTTrainer will use as the training text.
    """
    # Case 1: simple single-text field
    if text_field in ex and isinstance(ex[text_field], str) and ex[text_field].strip():
        return ex[text_field]

    # Case 2: instruction-style fields
    inst = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output", "")

    # If instruction schema is present, format a Narrator-style template
    if any([inst, inp, out]):
        # You can customize the style below to fit your DND Narrator tone.
        # The model learns to produce the "Assistant" part given the "System/User" context.
        parts = []
        parts.append("### System:\nYou are a vivid, cinematic DND Narrator. Keep style immersive and coherent.")
        if inst:
            parts.append(f"\n\n### User Instruction:\n{inst}")
        if inp:
            parts.append(f"\n\n### Context:\n{inp}")
        if out:
            parts.append(f"\n\n### Assistant (Narrator):\n{out}")
        return "".join(parts)

    # If nothing fits, return empty string (will be filtered out)
    return ""


def load_training_dataset(dataset_name: str, dataset_path: str, data_dir: str, text_field: str):
    """
    Load dataset from HF Hub (dataset_name), local file (dataset_path),
    or default path (/data/train.jsonl). Returns a datasets.Dataset.
    """
    if dataset_name:
        return load_dataset(dataset_name, split="train")

    if dataset_path:
        # auto-detect JSON/JSONL
        if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
            return load_dataset("json", data_files=dataset_path, split="train")
        else:
            raise ValueError("Unsupported dataset format. Use JSON/JSONL or pass --dataset_name.")

    # default
    default_path = os.path.join(data_dir, "train.jsonl")
    return load_dataset("json", data_files=default_path, split="train")


def main():
    parser = argparse.ArgumentParser("QLoRA Fine-tuning for DND Narrator")
    # Model & I/O
    parser.add_argument("--model_name", type=str, default=os.getenv("MODEL_NAME", "mistralai/Mistral-7B-v0.1"))
    parser.add_argument("--dataset_name", type=str, default=None, help="HF dataset name (e.g., tatsu-lab/alpaca).")
    parser.add_argument("--dataset_path", type=str, default=None, help="Local .json/.jsonl path.")
    parser.add_argument("--data_dir", type=str, default=os.getenv("DATA_DIR", "/data"))
    parser.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR", "/outputs"))
    parser.add_argument("--text_field", type=str, default="text", help="Text field name for single-field data.")

    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, default=int(os.getenv("MAX_STEPS", "200")))
    parser.add_argument("--per_device_train_batch_size", type=int, default=int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1")))
    parser.add_argument("--gradient_accumulation_steps", type=int, default=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8")))
    parser.add_argument("--learning_rate", type=float, default=float(os.getenv("LR", "2e-4")))
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--packing", action="store_true", help="Enable example packing for efficiency.")

    # Mode
    parser.add_argument("--train", action="store_true", help="Run QLoRA fine-tuning.")
    parser.add_argument("--chat", action="store_true", help="Run interactive narrator generation with the adapter.")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to saved LoRA adapter (for --chat).")

    args = parser.parse_args()

    # ------------------------ Common tokenizer / quant config ------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ----------------------------- TRAIN MODE ---------------------------------------
    if args.train:
        # Load base model in 4-bit
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # LoRA config (tuned for Causal LM; adjust target_modules per backbone)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none"
        )

        # Load dataset and map to single training text
        raw_ds = load_training_dataset(args.dataset_name, args.dataset_path, args.data_dir, args.text_field)

        def _map_fn(ex):
            text = build_text_from_example(ex, args.text_field)
            return {"text": text}

        ds = raw_ds.map(_map_fn)
        # filter out empty texts
        ds = ds.filter(lambda x: isinstance(x["text"], str) and len(x["text"].strip()) > 0)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            max_steps=args.max_steps,
            bf16=True,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            report_to="none"
        )

        # Trainer for SFT with LoRA/QLoRA
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=ds,
            peft_config=lora_config,
            dataset_text_field="text",
            packing=args.packing,
            max_seq_length=args.max_seq_length
        )

        print("[INFO] Start QLoRA training (Narrator)...")
        trainer.train()
        print("[INFO] Saving adapter and tokenizer...")
        adapter_path = os.path.join(args.output_dir, "qlora_adapter")
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[INFO] Done. Adapter saved to: {adapter_path}")

    # ------------------------------ CHAT MODE ---------------------------------------
    if args.chat:
        adapter_path = args.adapter_dir or os.path.join(args.output_dir, "qlora_adapter")
        # Load base model in 4-bit
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        # Attach LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        print("\n[INFO] Narrator chat mode. Type your DM prompt. Ctrl+C to exit.\n")
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        while True:
            try:
                user = input("DM> ").strip()
                if not user:
                    continue

                # Build a lightweight Narrator-style prompt for inference
                prompt = (
                    "### System:\nYou are a cinematic DND Narrator. "
                    "Write immersive, vivid, concise scene narration.\n\n"
                    f"### User Instruction:\n{user}\n\n"
                    "### Assistant (Narrator):\n"
                )

                input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
                gen_ids = model.generate(
                    **input_ids,
                    max_new_tokens=256,
                    temperature=0.9,
                    top_p=0.95,
                    do_sample=True,
                    streamer=streamer
                )
                # TextStreamer already prints tokens; you can also decode if needed.
                print("\n")

            except KeyboardInterrupt:
                print("\n[INFO] Exiting chat.")
                break
