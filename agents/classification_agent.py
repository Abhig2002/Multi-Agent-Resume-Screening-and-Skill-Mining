# stage 5 - tfidf+linearsvc (baseline) vs sbert+rf vs sbert+linearsvc (proposed)

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "evaluation" / "results"
STAGE_RESULTS_DIR = RESULTS_DIR / "stage5_results"

CLEAN_RESUMES_PATH = PROCESSED_DIR / "clean_resumes.csv"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"

RANDOM_STATE = 42


def stratified_three_way_indices(y: list, n_samples: int):
    idx = np.arange(n_samples)
    y_arr = np.array(y, dtype=object)
    idx_tv, idx_te, _, y_te = train_test_split(
        idx,
        y_arr,
        test_size=0.15,
        stratify=y_arr,
        random_state=RANDOM_STATE,
    )
    y_tv = y_arr[idx_tv]
    idx_tr, idx_va, _, _ = train_test_split(
        idx_tv,
        y_tv,
        test_size=0.15 / 0.85,
        stratify=y_tv,
        random_state=RANDOM_STATE,
    )
    return idx_tr, idx_va, idx_te


def run(sample_size: int | None = None) -> None:
    from evaluation.metrics import compute_classification_and_disparity

    STAGE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEAN_RESUMES_PATH)
    emb = np.load(EMBEDDINGS_PATH)

    if len(df) != emb.shape[0]:
        raise ValueError(
            f"Row mismatch: clean_resumes has {len(df)} rows, embeddings {emb.shape[0]}"
        )

    if sample_size:
        df = df.head(sample_size).copy()
        emb = emb[:sample_size]

    texts = df["clean_text"].fillna("").astype(str).tolist()
    y = df["Category"].astype(str).tolist()
    ids = df["ID"].astype(str).tolist()
    n = len(df)

    idx_tr, idx_va, idx_te = stratified_three_way_indices(y, n)

    Xt_text = [texts[i] for i in idx_tr]
    Xv_text = [texts[i] for i in idx_va]
    Xs_text = [texts[i] for i in idx_te]
    Xt_emb = emb[idx_tr]
    Xv_emb = emb[idx_va]
    Xs_emb = emb[idx_te]
    yt = [y[i] for i in idx_tr]
    yv = [y[i] for i in idx_va]
    ys = [y[i] for i in idx_te]
    test_ids = [ids[i] for i in idx_te]

    tfidf_clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), min_df=2)),
            (
                "clf",
                LinearSVC(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
            ),
        ]
    )
    tfidf_clf.fit(Xt_text, yt)
    pred_tfidf_val = tfidf_clf.predict(Xv_text)
    pred_tfidf_test = tfidf_clf.predict(Xs_text)

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(Xt_emb, yt)
    pred_rf_val = rf.predict(Xv_emb)
    pred_rf_test = rf.predict(Xs_emb)

    sbert_svc = LinearSVC(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
    sbert_svc.fit(Xt_emb, yt)
    pred_sbert_svc_val = sbert_svc.predict(Xv_emb)
    pred_sbert_svc_test = sbert_svc.predict(Xs_emb)

    metrics_tfidf_val = compute_classification_and_disparity(list(yv), list(pred_tfidf_val))
    metrics_tfidf_test = compute_classification_and_disparity(list(ys), list(pred_tfidf_test))
    metrics_rf_val = compute_classification_and_disparity(list(yv), list(pred_rf_val))
    metrics_rf_test = compute_classification_and_disparity(list(ys), list(pred_rf_test))
    metrics_sbert_svc_val = compute_classification_and_disparity(list(yv), list(pred_sbert_svc_val))
    metrics_sbert_svc_test = compute_classification_and_disparity(list(ys), list(pred_sbert_svc_test))

    def _row(model, split, m):
        return {
            "model": model,
            "split": split,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "disparity_gap": m["disparity"]["gap"],
        }

    comparison = pd.DataFrame(
        [
            _row("tfidf_linearsvc", "val",  metrics_tfidf_val),
            _row("tfidf_linearsvc", "test", metrics_tfidf_test),
            _row("sbert_rf",        "val",  metrics_rf_val),
            _row("sbert_rf",        "test", metrics_rf_test),
            _row("sbert_linearsvc", "val",  metrics_sbert_svc_val),
            _row("sbert_linearsvc", "test", metrics_sbert_svc_test),
        ]
    )
    comparison.to_csv(STAGE_RESULTS_DIR / "classification_comparison.csv", index=False)
    print("wrote", STAGE_RESULTS_DIR / "classification_comparison.csv")

    per_class_rows = []
    labels = sorted(set(ys))
    for model_name, pred in [
        ("tfidf_linearsvc", pred_tfidf_test),
        ("sbert_rf", pred_rf_test),
        ("sbert_linearsvc", pred_sbert_svc_test),
    ]:
        f1s = f1_score(ys, pred, labels=labels, average=None, zero_division=0)
        for lab, f1v in zip(labels, f1s):
            per_class_rows.append({"model": model_name, "category": lab, "f1": float(f1v)})
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(STAGE_RESULTS_DIR / "per_class_f1_disparity.csv", index=False)
    print("wrote", STAGE_RESULTS_DIR / "per_class_f1_disparity.csv")

    pred_detail = pd.DataFrame(
        {
            "ID": test_ids,
            "y_true": ys,
            "pred_tfidf": pred_tfidf_test,
            "pred_sbert_rf": pred_rf_test,
            "pred_sbert_linearsvc": pred_sbert_svc_test,
        }
    )
    pred_detail.to_csv(STAGE_RESULTS_DIR / "classification_test_predictions.csv", index=False)
    print("wrote", STAGE_RESULTS_DIR / "classification_test_predictions.csv")
    print("wrote stage5_results/")
    print("stage 5 done")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-size", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(sample_size=args.sample_size)
