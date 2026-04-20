# run stuff from project root: python run_pipeline.py --stage ...

import argparse

from agents.arm_agent import run as run_arm
from agents.classification_agent import run as run_classification
from agents.clustering_agent import run as run_clustering
from agents.matching_agent import run as run_matching
from agents.preprocessing_agent import run as run_preprocessing
from agents.skill_extraction_agent import run as run_skill_extraction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "preprocess",
            "skills",
            "week1",
            "cluster",
            "arm",
            "classify",
            "match",
            "week2",
        ],
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
    parser.add_argument(
        "--k-min",
        type=int,
        default=8,
        help="K-Means min k (cluster stage).",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=16,
        help="K-Means max k (cluster stage).",
    )
    parser.add_argument(
        "--normalize-embeddings",
        dest="normalize_embeddings",
        action="store_true",
        help="L2-normalize embeddings before K-Means.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        help="Disable L2-normalization before K-Means.",
    )
    parser.set_defaults(normalize_embeddings=True)
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.05,
        help="Apriori min_support (arm stage).",
    )
    parser.add_argument(
        "--match-top-k",
        type=int,
        default=10,
        help="Top-K resumes per job (match stage).",
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
    elif args.stage == "cluster":
        run_clustering(
            sample_size=args.sample_size,
            k_min=args.k_min,
            k_max=args.k_max,
            normalize_embeddings=args.normalize_embeddings,
        )
    elif args.stage == "arm":
        run_arm(
            min_support=args.min_support,
            sample_size=args.sample_size,
        )
    elif args.stage == "classify":
        run_classification(sample_size=args.sample_size)
    elif args.stage == "match":
        run_matching(
            sample_size=args.sample_size,
            top_k=args.match_top_k,
        )
    elif args.stage == "week2":
        run_clustering(
            sample_size=args.sample_size,
            k_min=args.k_min,
            k_max=args.k_max,
            normalize_embeddings=args.normalize_embeddings,
        )
        run_arm(min_support=args.min_support, sample_size=args.sample_size)
        run_classification(sample_size=args.sample_size)
        run_matching(sample_size=args.sample_size, top_k=args.match_top_k)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
