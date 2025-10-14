# scripts/run_all.py
import sys

from scripts.prepare_data import main as prepare
from scripts.train_sft import main as train
from scripts.eval_offline import main as evaluate
from scripts.register import main as register

def run():
    try:
        print("[run_all] step 1/4: prepare_data")
        prepare()
        print("[run_all] step 2/4: train_sft")
        train()
        print("[run_all] step 3/4: eval_offline")
        evaluate()
        print("[run_all] step 4/4: register")
        register()
        print("[run_all] DONE")
    except Exception as e:
        print(f"[run_all] FAILED: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    run()
