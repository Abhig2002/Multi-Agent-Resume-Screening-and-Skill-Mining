# metrics for classify + matching

import re
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def compute_classification_metrics(
    y_true: List,
    y_pred: List,
) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    labels = sorted(set(y_true) | set(y_pred))
    per_class_vals = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = dict(zip(labels, per_class_vals))

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "per_class_f1": per_class_f1,
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def compute_disparity_metrics(per_class_f1: Dict) -> Dict[str, float]:
    if not per_class_f1:
        return {"min_f1": 0.0, "max_f1": 0.0, "gap": 0.0, "std": 0.0}
    vals = np.array(list(per_class_f1.values()), dtype=float)
    return {
        "min_f1": float(np.min(vals)),
        "max_f1": float(np.max(vals)),
        "gap": float(np.max(vals) - np.min(vals)),
        "std": float(np.std(vals)),
    }


def compute_classification_and_disparity(y_true: List, y_pred: List) -> Dict[str, Any]:
    base = compute_classification_metrics(y_true, y_pred)
    base["disparity"] = compute_disparity_metrics(base["per_class_f1"])
    return base


def compare_models(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        row: Dict[str, Any] = {"model": name}
        for k, v in metrics.items():
            if k in ("per_class_f1", "report"):
                continue
            if k == "disparity" and isinstance(v, dict):
                for dk, dv in v.items():
                    row["disparity_" + dk] = dv
            elif isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if "macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False)
    return df


def compute_precision_at_k(ranked_ids: List, relevant_ids: Set, k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked_ids[:k]
    hits = sum(1 for rid in topk if rid in relevant_ids)
    return hits / k


def compute_mean_precision_at_k(
    all_ranked_ids: List[List],
    all_relevant_ids: List[Set],
    k: int,
) -> float:
    scores = [
        compute_precision_at_k(r, rel, k)
        for r, rel in zip(all_ranked_ids, all_relevant_ids)
    ]
    return float(np.mean(scores)) if scores else 0.0


def build_relevant_ids_by_category(resumes_df: pd.DataFrame, jd_category: str) -> Set:
    mask = resumes_df["Category"].astype(str) == str(jd_category)
    return set(resumes_df.loc[mask, "ID"].astype(str).tolist())


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def infer_categories_in_text(text: str, categories: List[str]) -> List[str]:
    normalized_text = _normalize_text(text)
    text_tokens = set(normalized_text.split())

    # Aliases help map real-world job wording to resume dataset categories.
    alias_by_category = {
        "INFORMATION-TECHNOLOGY": [
            "software developer",
            "software engineer",
            "web developer",
            "full stack",
            "frontend",
            "backend",
            "programmer",
            "python",
            "java",
            "dot net",
            "asp net",
            "devops",
            "cloud engineer",
        ],
        "BUSINESS-DEVELOPMENT": ["business development", "lead generation", "b2b sales"],
        "PUBLIC-RELATIONS": ["public relations", "media relations", "press release"],
        "DIGITAL-MEDIA": ["digital marketing", "digital media", "seo", "sem", "social media"],
        "HR": ["human resources", "talent acquisition", "recruiter", "hr executive"],
        "BPO": ["bpo", "call center", "voice process", "customer support"],
        "ACCOUNTANT": ["accountant", "bookkeeping", "tally", "audit", "gst"],
        "FINANCE": ["financial analyst", "finance", "fp a", "investment"],
        "SALES": ["sales executive", "inside sales", "field sales", "account manager"],
        "TEACHER": ["teacher", "tutor", "faculty", "lecturer"],
    }

    found = []
    for cat in categories:
        cat_str = str(cat).upper()
        variants = [
            _normalize_text(cat_str),
            _normalize_text(cat_str.replace("-", " ")),
            _normalize_text(cat_str.replace("-", "")),
        ] + [_normalize_text(a) for a in alias_by_category.get(cat_str, [])]

        for v in variants:
            if not v:
                continue
            if v in normalized_text:
                found.append(cat)
                break
            v_tokens = set(v.split())
            if v_tokens and v_tokens.issubset(text_tokens):
                found.append(cat)
                break

    return found
