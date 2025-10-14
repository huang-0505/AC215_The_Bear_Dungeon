from pathlib import Path
import json
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,  
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# gemma-2-2b-it
MODEL_NAME = "google/gemma-2-2b-it"

DATA_DIR = Path("/app/data/processed")
ART_DIR = Path("/app/artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = ART_DIR / "model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class TxtDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_len: int = 512):
        self.examples = [r["input"] + "\n" + r["output"] for r in rows]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)
        labels = ids.clone()
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def collate_fn(batch, pad_token_id):
    max_len = max(x["input_ids"].size(0) for x in batch)

    def pad(seq, pad_id):
        if seq.size(0) < max_len:
            pad_len = max_len - seq.size(0)
            seq = torch.cat([seq, torch.full((pad_len,), pad_id, dtype=seq.dtype)])
        return seq

    input_ids = torch.stack([pad(b["input_ids"], pad_token_id) for b in batch])
    attention_mask = torch.stack([pad(b["attention_mask"], 0) for b in batch])
    labels = torch.stack([pad(b["labels"], -100) for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate(model, loader):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            out = model(**batch)
            loss = out.loss
            tokens = batch["attention_mask"].sum().item()
            tot_loss += loss.item() * tokens
            tot_tok += tokens
    avg_loss = tot_loss / max(tot_tok, 1)
    try:
        ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    except Exception:
        ppl = None
    return {"val_loss": avg_loss, "val_ppl": ppl}


def main():
    # 1) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) 4-bit（BitsAndBytesConfig）
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,  
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 4) LoRA 
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"],  # 基于模型调整
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 5) data
    train_rows = _read_jsonl(DATA_DIR / "train.jsonl")
    val_rows = _read_jsonl(DATA_DIR / "val.jsonl")
    train_ds = TxtDataset(train_rows, tokenizer, max_len=512)
    val_ds = TxtDataset(val_rows, tokenizer, max_len=512)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    # 6) optim
    optim = AdamW(model.parameters(), lr=2e-4)

    # 7) train
    model.train()
    for step, batch in enumerate(train_loader, 1):
        for k in batch:
            batch[k] = batch[k].to(DEVICE)
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        if step % 5 == 0:
            print(f"[train] step={step}, loss={loss.item():.4f}")

    # 8) evaluate
    metrics = evaluate(model, val_loader)
    (ART_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"[train_sft][QLoRA] adapter saved to {MODEL_OUT}")
    print(f"[train_sft] metrics: {metrics}")


if __name__ == "__main__":
    main()
