# stage 6 - cosine match jobs to resumes

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "evaluation" / "results"
STAGE_RESULTS_DIR = RESULTS_DIR / "stage6_results"

CLEAN_RESUMES_PATH = PROCESSED_DIR / "clean_resumes.csv"
CLEAN_JDS_PATH = PROCESSED_DIR / "clean_jds.csv"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
SBERT_MODEL = "all-MiniLM-L6-v2"


def run(
    sample_size: int | None = None,
    top_k: int = 10,
    eval_precision_k: bool = True,
    category_rerank: bool = True,
    category_boost: float = 0.20,
    rerank_pool_size: int = 300,
) -> None:
    from evaluation.metrics import (
        compute_mean_precision_at_k,
        compute_precision_at_k,
        infer_categories_in_text,
    )

    STAGE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    resumes = pd.read_csv(CLEAN_RESUMES_PATH)
    emb_res = np.load(EMBEDDINGS_PATH)
    if len(resumes) != emb_res.shape[0]:
        raise ValueError("Resume rows != embedding rows")

    if not CLEAN_JDS_PATH.exists():
        print("need", CLEAN_JDS_PATH, "run preprocess first")
        return

    jds = pd.read_csv(CLEAN_JDS_PATH)
    if sample_size:
        jds = jds.head(sample_size).copy()

    texts_jd = jds["clean_text"].fillna("").astype(str).tolist()
    jd_ids = jds["JobID"].astype(str).tolist()

    print("encoding jobs...")
    model = SentenceTransformer(SBERT_MODEL)
    emb_jd = model.encode(texts_jd, show_progress_bar=True, convert_to_numpy=True)

    sim = cosine_similarity(emb_jd, emb_res)
    resume_ids = resumes["ID"].astype(str).tolist()
    resume_categories = resumes["Category"].astype(str).tolist()
    categories = sorted(resumes["Category"].astype(str).unique().tolist())

    rows = []
    all_ranked = []
    all_relevant = []

    for i, jid in enumerate(jd_ids):
        scores = sim[i].copy()
        jd_text = texts_jd[i]
        matched_cats = infer_categories_in_text(jd_text, categories)

        if category_rerank and matched_cats:
            matched_set = set(matched_cats)
            cat_mask = np.array([c in matched_set for c in resume_categories], dtype=float)

            # Semantic-first reranking:
            # 1) take top semantic pool by cosine
            # 2) apply a soft category bonus inside that pool
            # 3) keep remaining resumes in original cosine order
            base_order = np.argsort(-scores)
            pool_n = max(top_k, min(rerank_pool_size, len(base_order)))
            pool_idx = base_order[:pool_n]

            boosted_pool_scores = scores[pool_idx] + (category_boost * cat_mask[pool_idx])
            pool_order_local = np.argsort(-boosted_pool_scores)
            boosted_pool_idx = pool_idx[pool_order_local]
            tail_idx = base_order[pool_n:]
            order = np.concatenate([boosted_pool_idx, tail_idx])
        else:
            order = np.argsort(-scores)

        ranked = [resume_ids[j] for j in order[:top_k]]
        rows.append(
            {
                "JobID": jid,
                "ranked_resume_ids": json.dumps(ranked),
                "matched_categories": json.dumps(matched_cats),
                "category_rerank_applied": bool(category_rerank and len(matched_cats) > 0),
            }
        )

        if eval_precision_k:
            rel = set()
            for cat in matched_cats:
                mask = resumes["Category"].astype(str) == cat
                rel.update(resumes.loc[mask, "ID"].astype(str).tolist())
            full_ranked = [resume_ids[j] for j in order]
            all_ranked.append(full_ranked)
            all_relevant.append(rel)

    out_rank = pd.DataFrame(rows)
    out_rank.to_csv(STAGE_RESULTS_DIR / "matching_topk.csv", index=False)
    print("wrote", STAGE_RESULTS_DIR / "matching_topk.csv")

    if eval_precision_k and all_relevant:
        nonempty_pairs = [(r, rel) for r, rel in zip(all_ranked, all_relevant) if len(rel) > 0]
        summary = []
        for k in (5, 10):
            mean_p = compute_mean_precision_at_k(all_ranked, all_relevant, k)
            mean_p_nonempty = (
                compute_mean_precision_at_k(
                    [r for r, _ in nonempty_pairs],
                    [rel for _, rel in nonempty_pairs],
                    k,
                )
                if nonempty_pairs
                else 0.0
            )
            summary.append(
                {
                    "k": k,
                    "mean_precision_at_k": mean_p,
                    "mean_precision_at_k_nonempty": mean_p_nonempty,
                    "n_queries": len(all_ranked),
                    "n_queries_with_nonempty_relevant": len(nonempty_pairs),
                    "n_queries_with_empty_relevant": sum(1 for s in all_relevant if len(s) == 0),
                }
            )
        prec_df = pd.DataFrame(summary)
        prec_df.to_csv(STAGE_RESULTS_DIR / "matching_precision_at_k.csv", index=False)
        print("wrote matching_precision_at_k.csv (heuristic: category + alias keyword matching)")

    print("wrote stage6_results/")
    print("stage 6 done")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--no-precision-eval", action="store_true")
    p.add_argument("--no-category-rerank", action="store_true")
    p.add_argument("--category-boost", type=float, default=0.20)
    p.add_argument("--rerank-pool-size", type=int, default=300)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        sample_size=args.sample_size,
        top_k=args.top_k,
        eval_precision_k=not args.no_precision_eval,
        category_rerank=not args.no_category_rerank,
        category_boost=args.category_boost,
        rerank_pool_size=args.rerank_pool_size,
    )
