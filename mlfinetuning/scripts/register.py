# scripts/register.py
import json
from datetime import datetime
from pathlib import Path

ART_DIR = Path("/app/artifacts")
REG = ART_DIR / "model_registry.json"

def main():
    record = {
        "version": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "model_path": str(ART_DIR / "model"),
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    prev = []
    if REG.exists():
        try:
            prev = json.loads(REG.read_text(encoding="utf-8"))
        except Exception:
            prev = []
    prev.append(record)
    REG.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(f"[register] updated {REG}")

if __name__ == "__main__":
    main()
