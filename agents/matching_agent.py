# stage 6 - cosine match jobs to resumes (no llm rerank in this version)

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
) -> None:
    from evaluation.metrics import (
        compute_mean_precision_at_k,
        compute_precision_at_k,
        infer_categories_in_text,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
    categories = sorted(resumes["Category"].astype(str).unique().tolist())

    rows = []
    all_ranked = []
    all_relevant = []

    for i, jid in enumerate(jd_ids):
        scores = sim[i]
        order = np.argsort(-scores)
        ranked = [resume_ids[j] for j in order[:top_k]]
        rows.append({"JobID": jid, "ranked_resume_ids": json.dumps(ranked)})

        if eval_precision_k:
            jd_text = texts_jd[i]
            matched_cats = infer_categories_in_text(jd_text, categories)
            rel = set()
            for cat in matched_cats:
                mask = resumes["Category"].astype(str) == cat
                rel.update(resumes.loc[mask, "ID"].astype(str).tolist())
            full_ranked = [resume_ids[j] for j in order]
            all_ranked.append(full_ranked)
            all_relevant.append(rel)

    out_rank = pd.DataFrame(rows)
    out_rank.to_csv(RESULTS_DIR / "matching_topk.csv", index=False)
    out_rank.to_csv(STAGE_RESULTS_DIR / "matching_topk.csv", index=False)
    print("wrote matching_topk.csv")

    if eval_precision_k and all_relevant:
        summary = []
        for k in (5, 10):
            mean_p = compute_mean_precision_at_k(all_ranked, all_relevant, k)
            summary.append(
                {
                    "k": k,
                    "mean_precision_at_k": mean_p,
                    "n_queries": len(all_ranked),
                    "n_queries_with_empty_relevant": sum(1 for s in all_relevant if len(s) == 0),
                }
            )
        prec_df = pd.DataFrame(summary)
        prec_df.to_csv(RESULTS_DIR / "matching_precision_at_k.csv", index=False)
        prec_df.to_csv(STAGE_RESULTS_DIR / "matching_precision_at_k.csv", index=False)
        print("wrote matching_precision_at_k.csv (heuristic: category substring in jd text)")

    print("wrote stage6_results/")
    print("stage 6 done")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--no-precision-eval", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        sample_size=args.sample_size,
        top_k=args.top_k,
        eval_precision_k=not args.no_precision_eval,
    )
