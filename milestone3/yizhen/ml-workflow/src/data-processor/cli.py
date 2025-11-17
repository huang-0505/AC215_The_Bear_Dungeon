import argparse
import cleanser


def main(args=None):
    print("CLI Arguments:", args)

    if args.clean:
        print("▶️  Cleaning DnD dataset (explicit flag)...")
        cleanser.run_cleaning()
        return

    print("💡 No flags provided → default action: cleaning dataset")
    cleanser.run_cleaning()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DnD Data Processor CLI - Clean dataset for Gemini fine-tuning"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean raw DnD dataset and upload cleaned files to GCS",
    )

    args = parser.parse_args()
    main(args)
