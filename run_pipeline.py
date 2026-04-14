"""
Project entrypoint (template)
=============================
Use this script to run stages quickly from the repo root.
"""

from __future__ import annotations

import argparse

from agents.preprocessing_agent import run as run_preprocessing
from agents.skill_extraction_agent import run as run_skill_extraction


def parse_args():
    parser = argparse.ArgumentParser(description="Run project pipeline stages.")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "skills", "week1"],
        required=True,
        help="Which stage to run.",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only relevant for --stage skills/week1.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage == "preprocess":
        run_preprocessing(sample_size=args.sample_size)
    elif args.stage == "skills":
        run_skill_extraction(sample_size=args.sample_size, skip_llm=args.skip_llm)
    elif args.stage == "week1":
        run_preprocessing(sample_size=args.sample_size)
        run_skill_extraction(sample_size=args.sample_size, skip_llm=args.skip_llm)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
