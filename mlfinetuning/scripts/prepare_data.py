# scripts/prepare_data.py
# generate the train and val data, after we have train.jsonl / val.jsonl we will abandon this one
import json
from pathlib import Path

def main():
    data_dir = Path("/app/data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    train_samples = [
        {"input": "USER: We enter the tavern\nSYS: You are a narrator DM.",
         "output": "The tavern is dimly lit. A bard plays softly."},
        {"input": "USER: I talk to the bartender\nSYS: Keep answers concise.",
         "output": "\"What will it be?\" the bartender asks, polishing a glass."},
    ]
    val_samples = [
        {"input": "USER: I look for quests\nSYS: Keep answers concise.",
         "output": "A faded notice board lists a missing caravan."}
    ]

    with train_path.open("w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[prepare_data] wrote {train_path} and {val_path}")

if __name__ == "__main__":
    main()
