"""
Project entrypoint (template)
=============================
Use this script to run stages quickly from the repo root.
"""

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
    parser.add_argument(
        "--llm-provider",
        choices=["groq", "huggingface"],
        default="groq",
        help="LLM backend for skill extraction stage.",
    )
    parser.add_argument(
        "--hf-model-name",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Hugging Face model id when --llm-provider huggingface.",
    )
    parser.add_argument(
        "--hf-max-new-tokens",
        type=int,
        default=220,
        help="Max generated tokens for Hugging Face extraction.",
    )
    parser.add_argument(
        "--refill-empty",
        action="store_true",
        help="Re-run only cached skill rows where technical and soft are both empty.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage == "preprocess":
        run_preprocessing(sample_size=args.sample_size)
    elif args.stage == "skills":
        run_skill_extraction(
            sample_size=args.sample_size,
            skip_llm=args.skip_llm,
            llm_provider=args.llm_provider,
            hf_model_name=args.hf_model_name,
            hf_max_new_tokens=args.hf_max_new_tokens,
            refill_empty=args.refill_empty,
        )
    elif args.stage == "week1":
        run_preprocessing(sample_size=args.sample_size)
        run_skill_extraction(
            sample_size=args.sample_size,
            skip_llm=args.skip_llm,
            llm_provider=args.llm_provider,
            hf_model_name=args.hf_model_name,
            hf_max_new_tokens=args.hf_max_new_tokens,
            refill_empty=args.refill_empty,
        )
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
