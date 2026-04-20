# stage 4 - apriori on skills

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "evaluation" / "results"
STAGE_RESULTS_DIR = RESULTS_DIR / "stage4_results"

SKILLS_PATH = PROCESSED_DIR / "skill_lists.json"
CLEAN_RESUMES_PATH = PROCESSED_DIR / "clean_resumes.csv"


def load_transactions(skills_path: Path, id_order: list | None = None) -> list:
    with skills_path.open(encoding="utf-8") as f:
        data = json.load(f)
    transactions = []
    if id_order is not None:
        keys = [str(x) for x in id_order]
    else:
        def _key(r):
            try:
                return int(r)
            except (ValueError, TypeError):
                return str(r)

        keys = sorted(data.keys(), key=_key)

    for rid in keys:
        entry = data.get(rid)
        if entry is None:
            transactions.append([])
            continue
        tech = entry.get("technical", []) or []
        soft = entry.get("soft", []) or []
        items = [str(x).strip().lower() for x in tech + soft if str(x).strip()]
        transactions.append(items)
    return transactions


def run(
    min_support: float = 0.05,
    min_threshold: float = 0.5,
    metric: str = "confidence",
    sample_size: int | None = None,
) -> None:
    STAGE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    id_order = None
    if CLEAN_RESUMES_PATH.exists():
        id_order = pd.read_csv(CLEAN_RESUMES_PATH)["ID"].astype(str).tolist()
    transactions = load_transactions(SKILLS_PATH, id_order=id_order)
    if sample_size:
        transactions = transactions[:sample_size]

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    oht = pd.DataFrame(te_ary, columns=te.columns_)

    frequent = apriori(oht, min_support=min_support, use_colnames=True, verbose=0)
    if frequent.empty:
        print("no frequent itemsets, try lower min_support")
        frequent.to_csv(STAGE_RESULTS_DIR / "frequent_itemsets.csv", index=False)
        return

    rules = association_rules(
        frequent,
        metric=metric,
        min_threshold=min_threshold,
    )
    rules = rules.sort_values("lift", ascending=False)
    rules.to_csv(STAGE_RESULTS_DIR / "association_rules.csv", index=False)
    frequent.to_csv(STAGE_RESULTS_DIR / "frequent_itemsets.csv", index=False)
    print(len(rules), "rules ->", STAGE_RESULTS_DIR / "association_rules.csv")

    meta = {
        "n_transactions": len(transactions),
        "n_unique_skills": int(oht.shape[1]),
        "min_support": min_support,
        "metric": metric,
        "min_threshold": min_threshold,
    }
    with (STAGE_RESULTS_DIR / "skill_item_matrix_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("wrote stage4_results/")
    print("stage 4 done")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min-support", type=float, default=0.05)
    p.add_argument("--min-threshold", type=float, default=0.5)
    p.add_argument("--metric", type=str, default="confidence")
    p.add_argument("--sample-size", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        min_support=args.min_support,
        min_threshold=args.min_threshold,
        metric=args.metric,
        sample_size=args.sample_size,
    )
