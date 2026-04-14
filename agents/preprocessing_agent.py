"""
Stage 1 - Preprocessing Agent (Template)
========================================
Fast scaffold so you can learn-by-implementing.

What this file gives you:
- End-to-end runnable skeleton for cleaning resume + job description text
- Small sample run mode (`--sample-size`) for quick iteration
- Clear TODO markers where you can customize behavior
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_RESUME_PATH = RAW_DIR / "Resume.csv"
RAW_JD_PATH = RAW_DIR / "job_dataset.csv"
OUT_CLEAN_RESUME_PATH = PROCESSED_DIR / "clean_resumes.csv"
OUT_CLEAN_JD_PATH = PROCESSED_DIR / "clean_jds.csv"


def setup_nlp():
    """Download/initialize NLP resources (safe to call repeatedly)."""
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    stops = set(stopwords.words("english"))
    return nlp, stops


def remove_html(text: str) -> str:
    """Remove HTML tags and collapse spaces."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """Lowercase and keep alphabetic characters/spaces only."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str, nlp, stop_words: set[str]) -> str:
    """
    Full cleaning pipeline for one document.

    TODO (learning):
    - Try preserving numbers like 'aws s3' or versions if you need them.
    - Try a custom stopword list for domain terms.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = remove_html(text)
    text = normalize_text(text)
    if not text:
        return ""

    doc = nlp(text)
    tokens = [t.lemma_.strip() for t in doc if t.lemma_ and t.lemma_ not in stop_words]
    tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)


def _find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def preprocess_resumes(df: pd.DataFrame, nlp, stop_words: set[str]) -> pd.DataFrame:
    """Create clean resume dataframe with stable columns for downstream stages."""
    resume_col = _find_column(df.columns, ["Resume_str", "resume", "resume_text", "text"])
    id_col = _find_column(df.columns, ["ID", "id"])
    category_col = _find_column(df.columns, ["Category", "category", "label"])

    if resume_col is None:
        raise ValueError("Could not find resume text column in Resume.csv")

    out = pd.DataFrame()
    out["ID"] = df[id_col] if id_col else range(len(df))
    out["Category"] = df[category_col] if category_col else "unknown"
    out["raw_text"] = df[resume_col].fillna("")
    out["clean_text"] = out["raw_text"].map(lambda x: clean_text(x, nlp, stop_words))
    return out


def preprocess_job_descriptions(df: pd.DataFrame, nlp, stop_words: set[str]) -> pd.DataFrame:
    """
    Build a single `raw_text` field from common job dataset columns.

    TODO (learning):
    - Experiment with different column combinations / weighting.
    """
    out = pd.DataFrame()
    out["JobID"] = df[_find_column(df.columns, ["JobID", "id"]) or df.columns[0]]
    title_col = _find_column(df.columns, ["Title"])
    skills_col = _find_column(df.columns, ["Skills"])
    resp_col = _find_column(df.columns, ["Responsibilities", "Responsibility"])
    keywords_col = _find_column(df.columns, ["Keywords", "Keyword"])

    text_parts = []
    for col in [title_col, skills_col, resp_col, keywords_col]:
        if col is not None:
            text_parts.append(df[col].fillna("").astype(str))

    if not text_parts:
        raise ValueError("No expected text columns found in job_dataset.csv")

    raw_text = text_parts[0]
    for part in text_parts[1:]:
        raw_text = raw_text + " " + part

    out["raw_text"] = raw_text
    out["clean_text"] = out["raw_text"].map(lambda x: clean_text(x, nlp, stop_words))
    return out


def run(sample_size: int | None = None) -> None:
    """Main Stage 1 execution."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    nlp, stop_words = setup_nlp()

    resumes = pd.read_csv(RAW_RESUME_PATH)
    jobs = pd.read_csv(RAW_JD_PATH)

    if sample_size:
        resumes = resumes.head(sample_size).copy()
        jobs = jobs.head(sample_size).copy()

    clean_resumes = preprocess_resumes(resumes, nlp, stop_words)
    clean_jobs = preprocess_job_descriptions(jobs, nlp, stop_words)

    clean_resumes.to_csv(OUT_CLEAN_RESUME_PATH, index=False)
    clean_jobs.to_csv(OUT_CLEAN_JD_PATH, index=False)

    print(f"Saved: {OUT_CLEAN_RESUME_PATH}")
    print(f"Saved: {OUT_CLEAN_JD_PATH}")
    print("Stage 1 complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 1 preprocessing agent.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="If provided, only process first N rows for fast testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(sample_size=args.sample_size)
