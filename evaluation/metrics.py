"""
Evaluation helpers (template)
=============================
Shared metric functions for Stage 5 and Stage 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def compute_classification_metrics(y_true: list, y_pred: list) -> dict:
    """
    Return core classification metrics.
    """
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


def compute_disparity_metrics(per_class_f1: dict) -> dict:
    """
    Category-level disparity (not demographic fairness).
    """
    if not per_class_f1:
        return {"min_f1": 0.0, "max_f1": 0.0, "gap": 0.0, "std": 0.0}
    vals = np.array(list(per_class_f1.values()), dtype=float)
    return {
        "min_f1": float(np.min(vals)),
        "max_f1": float(np.max(vals)),
        "gap": float(np.max(vals) - np.min(vals)),
        "std": float(np.std(vals)),
    }


def compute_classification_and_disparity(y_true: list, y_pred: list) -> dict:
    base = compute_classification_metrics(y_true, y_pred)
    base["disparity"] = compute_disparity_metrics(base["per_class_f1"])
    return base


def compare_models(results: dict) -> pd.DataFrame:
    rows = [{"model": name, **metrics} for name, metrics in results.items()]
    df = pd.DataFrame(rows)
    if "macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False)
    return df


def compute_precision_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked_ids[:k]
    hits = sum(1 for rid in topk if rid in relevant_ids)
    return hits / k


def compute_mean_precision_at_k(all_ranked_ids: list, all_relevant_ids: list, k: int) -> float:
    scores = [compute_precision_at_k(r, rel, k) for r, rel in zip(all_ranked_ids, all_relevant_ids)]
    return float(np.mean(scores)) if scores else 0.0
"""
Evaluation Metrics
==================
Purpose:
    Shared metric computation utilities used in Stage 5 (Classification)
    and Stage 6 (Job Matching). Centralizing metrics avoids duplication
    and makes it easy to swap metric implementations across agents.

Metrics Implemented:
    Classification:
        - Accuracy
        - Macro-F1 (primary metric — fair to minority classes)
        - Per-class F1 (useful for identifying hard categories)

    Ranking (Job Matching):
        - Precision@K — fraction of top-K results that are relevant
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: list,
    y_pred: list,
    label_names: list = None,
) -> dict:
    """
    Compute accuracy, macro-F1, and per-class F1 scores.

    Learning note — Macro-F1 vs Weighted-F1:
        Macro-F1:    Average F1 over all classes equally.
                     A class with 50 samples counts as much as one with 500.
                     Use this when all classes matter equally (our case: 24 job types).
        Weighted-F1: Average F1 weighted by class support (# samples).
                     Large classes dominate. Use for imbalanced tasks where
                     majority class performance matters most.

    Args:
        y_true:      List of ground-truth labels.
        y_pred:      List of predicted labels.
        label_names: Optional ordered list of class names for the report.

    Returns:
        dict with keys:
            "accuracy"    : float
            "macro_f1"    : float
            "weighted_f1" : float
            "per_class_f1": dict mapping class name → F1 score
            "report"      : full classification_report string
    """
    # TODO: acc = accuracy_score(y_true, y_pred)
    raise NotImplementedError("Compute accuracy with sklearn accuracy_score")

    # TODO: macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    raise NotImplementedError("Compute macro F1 score")

    # TODO: weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    raise NotImplementedError("Compute weighted F1 score")

    # TODO: report = classification_report(y_true, y_pred,
    #                                      target_names=label_names, zero_division=0)
    raise NotImplementedError("Generate full classification report string")

    # TODO: Compute per-class F1 scores:
    #   labels = sorted(set(y_true))
    #   per_class = f1_score(y_true, y_pred, labels=labels,
    #                        average=None, zero_division=0)
    #   per_class_f1 = dict(zip(labels, per_class))
    raise NotImplementedError("Compute per-class F1 and build dict")

    # TODO: return {
    #     "accuracy":     acc,
    #     "macro_f1":     macro_f1,
    #     "weighted_f1":  weighted_f1,
    #     "per_class_f1": per_class_f1,
    #     "report":       report,
    # }
    raise NotImplementedError("Return metrics dict")


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    label_names: list = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (14, 12),
) -> None:
    """
    Plot a normalized confusion matrix using seaborn heatmap.

    Learning note — Reading a Confusion Matrix:
        Rows = true labels, Columns = predicted labels.
        Diagonal = correct predictions (want these to be high).
        Off-diagonal = errors. The confusion matrix reveals which categories
        the model confuses with each other (e.g., "Java Developer" vs
        "Python Developer" — both are software roles).

    Args:
        y_true:      True labels.
        y_pred:      Predicted labels.
        label_names: Class names for axis ticks.
        title:       Plot title.
        figsize:     Figure size tuple.
    """
    # TODO: cm = confusion_matrix(y_true, y_pred, labels=label_names)
    #       cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalize by row
    raise NotImplementedError("Compute and normalize the confusion matrix")

    # TODO: Create a seaborn heatmap:
    #   fig, ax = plt.subplots(figsize=figsize)
    #   sns.heatmap(cm_norm, annot=False, fmt='.2f', xticklabels=label_names,
    #               yticklabels=label_names, ax=ax)
    #   ax.set_title(title)
    #   ax.set_xlabel('Predicted')
    #   ax.set_ylabel('True')
    #   plt.tight_layout()
    #   plt.show()
    raise NotImplementedError("Create seaborn heatmap of normalized confusion matrix")


def compare_models(results: dict) -> pd.DataFrame:
    """
    Create a comparison table across multiple models.

    Args:
        results: Dict mapping model_name → {"accuracy": float, "macro_f1": float, ...}

    Returns:
        DataFrame sorted by macro_f1 descending.
    """
    # TODO: rows = [{"model": name, **metrics} for name, metrics in results.items()]
    #       df = pd.DataFrame(rows).sort_values('macro_f1', ascending=False)
    #       return df
    raise NotImplementedError("Build and return sorted comparison DataFrame")


# ---------------------------------------------------------------------------
# Ranking metrics (Job Matching)
# ---------------------------------------------------------------------------

def compute_precision_at_k(
    ranked_resume_ids: list,
    relevant_resume_ids: set,
    k: int,
) -> float:
    """
    Compute Precision@K for a single query (one job description).

    Learning note — Precision@K:
        Precision@K = (# relevant items in top-K) / K

        Example:
            top-10 returned: [3, 7, 1, 44, 2, 88, 5, 9, 12, 30]
            relevant (same category): {3, 7, 44, 5, 12}
            relevant in top-10: {3, 7, 44, 5, 12} → 5 hits
            Precision@10 = 5/10 = 0.50

    Args:
        ranked_resume_ids:  Ordered list of resume IDs returned by the matcher.
        relevant_resume_ids: Set of resume IDs that are truly relevant (ground truth).
        k:                   Cut-off rank.

    Returns:
        Float in [0.0, 1.0].
    """
    # TODO: top_k = ranked_resume_ids[:k]
    #       hits = sum(1 for rid in top_k if rid in relevant_resume_ids)
    #       return hits / k
    raise NotImplementedError("Compute Precision@K")


def compute_mean_precision_at_k(
    all_ranked_ids: list,
    all_relevant_ids: list,
    k: int,
) -> float:
    """
    Compute mean Precision@K over multiple queries (multiple job descriptions).

    Args:
        all_ranked_ids:   List of ranked ID lists, one per query.
        all_relevant_ids: List of relevant ID sets, one per query.
        k:                Cut-off rank.

    Returns:
        Mean Precision@K across all queries.
    """
    # TODO: scores = [compute_precision_at_k(ranked, relevant, k)
    #                 for ranked, relevant in zip(all_ranked_ids, all_relevant_ids)]
    #       return np.mean(scores)
    raise NotImplementedError("Compute and return mean Precision@K across all queries")


def build_relevant_ids_by_category(
    resumes_df: pd.DataFrame,
    jd_category: str,
) -> set:
    """
    Find all resume IDs whose Category matches the job's expected category.

    This is the ground truth for Precision@K evaluation.

    Args:
        resumes_df:   DataFrame with columns ['ID', 'Category'].
        jd_category:  The category string for the target job description.

    Returns:
        Set of resume IDs that belong to jd_category.
    """
    # TODO: mask = resumes_df['Category'] == jd_category
    #       return set(resumes_df.loc[mask, 'ID'].tolist())
    raise NotImplementedError("Filter resumes by category and return set of IDs")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(results: dict) -> None:
    """
    Print a formatted comparison table to stdout.

    Args:
        results: Dict mapping model_name → {"accuracy": float, "macro_f1": float}
    """
    # TODO: Use compare_models() to build the DataFrame, then print it nicely.
    #   df = compare_models(results)
    #   print("\n=== Model Comparison ===")
    #   print(df[['model', 'accuracy', 'macro_f1']].to_string(index=False))
    raise NotImplementedError("Print comparison table using compare_models()")
