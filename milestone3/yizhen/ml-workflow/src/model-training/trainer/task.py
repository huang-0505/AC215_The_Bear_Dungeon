# trainer/task.py

import argparse

from trainer.gemini_tuner import (
    start_tuning,
    check_status,
    wait_until_complete,
    get_tuned_model_name,
)


def cmd_tune(args: argparse.Namespace) -> None:
    epochs = args.epochs
    print(f"=== COMMAND: START TUNING (epochs={epochs}) ===")
    job_name = start_tuning(epochs=epochs)
    print(f"✓ Tuning job launched: {job_name}")


def cmd_status(args: argparse.Namespace) -> None:
    print("=== COMMAND: CHECK STATUS ===")
    status = check_status()
    print(f"✓ Current status: {status}")


def cmd_wait(args: argparse.Namespace) -> None:
    print("=== COMMAND: WAIT FOR COMPLETION ===")
    final_status = wait_until_complete()
    print(f"✓ Job finished with status: {final_status}")


def cmd_get_model(args: argparse.Namespace) -> None:
    print("=== COMMAND: GET TUNED MODEL ===")
    name = get_tuned_model_name()
    print(f"✓ Tuned model resource name: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini Fine-Tuning Task Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- tune ----
    tune_parser = subparsers.add_parser("tune", help="Start a new tuning job")
    tune_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    tune_parser.set_defaults(func=cmd_tune)

    # ---- status ----
    status_parser = subparsers.add_parser(
        "status",
        help="Check tuning job status",
    )
    status_parser.set_defaults(func=cmd_status)

    # ---- wait ----
    wait_parser = subparsers.add_parser(
        "wait",
        help="Wait until tuning job finishes",
    )
    wait_parser.set_defaults(func=cmd_wait)

    # ---- get-model ----
    get_model_parser = subparsers.add_parser(
        "get-model",
        help="Print tuned model resource name",
    )
    get_model_parser.set_defaults(func=cmd_get_model)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
