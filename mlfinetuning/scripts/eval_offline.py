# scripts/eval_offline.py
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

ART_DIR = Path("/app/artifacts")
MODEL_DIR = ART_DIR / "model"
REPORT = ART_DIR / "eval_report.json"

def generate(prompt, tokenizer, model, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR))

    samples = [
        "USER: We enter the tavern\nSYS: You are a narrator DM.",
        "USER: I look for quests\nSYS: Keep answers concise.",
    ]
    gens = []
    for s in samples:
        out = generate(s, tokenizer, model, max_new_tokens=40)
        gens.append({"input": s, "output": out})

    REPORT.write_text(json.dumps({"samples": gens}, indent=2), encoding="utf-8")
    print(f"[eval_offline] wrote {REPORT}")

if __name__ == "__main__":
    main()
