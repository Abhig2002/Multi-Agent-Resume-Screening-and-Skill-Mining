# stage 3 - kmeans on resume embeddings

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "evaluation" / "results"
STAGE_RESULTS_DIR = RESULTS_DIR / "stage3_results"

CLEAN_RESUMES_PATH = PROCESSED_DIR / "clean_resumes.csv"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"

RANDOM_STATE = 42


def run(
    sample_size: int | None = None,
    k_min: int = 8,
    k_max: int = 16,
    normalize_embeddings: bool = True,
    pca_components: int | None = None,
) -> None:
    STAGE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEAN_RESUMES_PATH)
    emb = np.load(EMBEDDINGS_PATH)
    if len(df) != emb.shape[0]:
        raise ValueError(f"Row mismatch: {len(df)} vs {emb.shape[0]}")

    if sample_size:
        df = df.head(sample_size).copy()
        emb = emb[:sample_size]

    if normalize_embeddings:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        emb = emb / norms

    pca_variance_explained = None
    if pca_components is not None:
        n_components = min(pca_components, emb.shape[0], emb.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        emb = pca.fit_transform(emb)
        pca_variance_explained = float(pca.explained_variance_ratio_.sum())
        print(f"PCA: {n_components} components, {pca_variance_explained:.3f} variance explained")

    best_k = k_min
    best_sil = -1.0
    scores = {}
    inertias = {}
    db_scores = {}
    ch_scores = {}

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(emb)
        sil = silhouette_score(emb, labels, sample_size=min(2000, len(emb)), random_state=RANDOM_STATE)
        scores[k] = float(sil)
        inertias[k] = float(km.inertia_)
        db_scores[k] = float(davies_bouldin_score(emb, labels))
        ch_scores[k] = float(calinski_harabasz_score(emb, labels))
        if sil > best_sil:
            best_sil = sil
            best_k = k

    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_ids = km_final.fit_predict(emb)
    cluster_counts = pd.Series(cluster_ids).value_counts().sort_index()
    cluster_size_by_id = {str(int(k)): int(v) for k, v in cluster_counts.items()}

    categories = df["Category"].astype(str)
    nmi_with_category = float(
        normalized_mutual_info_score(categories.to_numpy(), cluster_ids)
    )
    purity_numer = pd.crosstab(cluster_ids, categories).max(axis=1).sum()
    purity_with_category = float(purity_numer / len(df))

    out = pd.DataFrame(
        {
            "ID": df["ID"].astype(str),
            "Category": df["Category"].astype(str),
            "cluster_id": cluster_ids,
        }
    )
    out.to_csv(STAGE_RESULTS_DIR / "cluster_assignments.csv", index=False)
    print("wrote clusters k=", best_k, "->", STAGE_RESULTS_DIR / "cluster_assignments.csv")

    sizes = list(cluster_size_by_id.values())
    summary = {
        "best_k": best_k,
        "best_silhouette": best_sil,
        "davies_bouldin_at_best_k": db_scores[best_k],
        "calinski_harabasz_at_best_k": ch_scores[best_k],
        "silhouette_by_k": scores,
        "davies_bouldin_by_k": db_scores,
        "calinski_harabasz_by_k": ch_scores,
        "inertia_by_k": inertias,
        "n_samples": int(len(df)),
        "normalized_embeddings": normalize_embeddings,
        "pca_components": pca_components,
        "pca_variance_explained": pca_variance_explained,
        "cluster_size_by_id": cluster_size_by_id,
        "cluster_size_min": int(min(sizes)),
        "cluster_size_max": int(max(sizes)),
        "cluster_size_mean": round(sum(sizes) / len(sizes), 1),
        "nmi_with_category": nmi_with_category,
        "purity_with_category": purity_with_category,
    }
    variant = "norm" if normalize_embeddings else "nonorm"
    if pca_components is not None:
        variant = f"{variant}_pca{pca_components}"
    stage_summary_path = STAGE_RESULTS_DIR / f"clustering_summary_{variant}.json"
    with stage_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (STAGE_RESULTS_DIR / "clustering_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote stage3_results/clustering_summary.json +", stage_summary_path.name)
    print("stage 3 done")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--k-min", type=int, default=8)
    p.add_argument("--k-max", type=int, default=16)
    p.add_argument(
        "--normalize-embeddings",
        dest="normalize_embeddings",
        action="store_true",
        help="L2-normalize each embedding before clustering.",
    )
    p.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        help="Disable L2-normalization before clustering.",
    )
    p.set_defaults(normalize_embeddings=True)
    p.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Reduce embeddings with PCA before clustering (e.g. 64 or 128). Default: no PCA.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        sample_size=args.sample_size,
        k_min=args.k_min,
        k_max=args.k_max,
        normalize_embeddings=args.normalize_embeddings,
        pca_components=args.pca_components,
    )
