"""
Utility to login to Hugging Face and download a model snapshot locally.

Usage:
    python scripts/download_model.py --model-id meta-llama/Llama-3.1-8B-Instruct --output-dir ./models_downloaded

Requires HUGGINGFACE_HUB_TOKEN in the environment or --token flag.
"""

import argparse
import os
from huggingface_hub import login, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot locally.")
    parser.add_argument("--model-id", required=True, help="Model repo id, e.g., meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", required=True, help="Destination directory for the snapshot")
    parser.add_argument("--revision", default="main", help="Branch/revision to download (default: main)")
    parser.add_argument("--token", default=None, help="HF token (fallback to HUGGINGFACE_HUB_TOKEN env)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HUGGINGFACE_HUB_TOKEN not set and --token not provided.")

    login(token=token)
    local_dir = snapshot_download(
        repo_id=args.model_id,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
        revision=args.revision,
    )
    print(f"Model downloaded to: {local_dir}")


if __name__ == "__main__":
    main()
