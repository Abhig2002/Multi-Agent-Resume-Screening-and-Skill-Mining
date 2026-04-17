"""
Stage 1 - Preprocessing Agent
==============================
Cleans resume and job description text using NLTK only (no spaCy dependency).
Compatible with Python 3.6+.
"""

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_RESUME_PATH = RAW_DIR / "Resume.csv"
RAW_JD_PATH = RAW_DIR / "job_dataset.csv"
OUT_CLEAN_RESUME_PATH = PROCESSED_DIR / "clean_resumes.csv"
OUT_CLEAN_JD_PATH = PROCESSED_DIR / "clean_jds.csv"


def setup_nlp():
    """Download NLTK resources and return lemmatizer + stopwords."""
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    lemmatizer = WordNetLemmatizer()
    stops = set(stopwords.words("english"))
    return lemmatizer, stops


def remove_html(text):
    # type: (str) -> str
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text):
    # type: (str) -> str
    """Lowercase and keep alphabetic characters/spaces only."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text, lemmatizer, stop_words):
    # type: (str, WordNetLemmatizer, Set[str]) -> str
    """Full cleaning pipeline for one document using NLTK."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = remove_html(text)
    text = normalize_text(text)
    if not text:
        return ""

    # Whitespace split only — normalize_text already strips non-letters, so we do not
    # need punkt / punkt_tab (NLTK 3.9+ moved tokenizer data and breaks old setups).
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 1
    ]
    return " ".join(tokens)


def _find_column(columns, candidates):
    # type: (Iterable[str], List[str]) -> Optional[str]
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def preprocess_resumes(df, lemmatizer, stop_words):
    # type: (pd.DataFrame, WordNetLemmatizer, Set[str]) -> pd.DataFrame
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
    print("Cleaning resumes...")
    out["clean_text"] = out["raw_text"].map(lambda x: clean_text(x, lemmatizer, stop_words))
    return out


def preprocess_job_descriptions(df, lemmatizer, stop_words):
    # type: (pd.DataFrame, WordNetLemmatizer, Set[str]) -> pd.DataFrame
    """Build a single clean_text field from common job dataset columns."""
    out = pd.DataFrame()
    id_col = _find_column(df.columns, ["JobID", "id"])
    out["JobID"] = df[id_col] if id_col else range(len(df))

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
    print("Cleaning job descriptions...")
    out["clean_text"] = out["raw_text"].map(lambda x: clean_text(x, lemmatizer, stop_words))
    return out


def run(sample_size=None):
    # type: (Optional[int]) -> None
    """Main Stage 1 execution."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    lemmatizer, stop_words = setup_nlp()

    resumes = pd.read_csv(RAW_RESUME_PATH)

    if sample_size:
        resumes = resumes.head(sample_size).copy()

    clean_resumes = preprocess_resumes(resumes, lemmatizer, stop_words)
    clean_resumes.to_csv(OUT_CLEAN_RESUME_PATH, index=False)
    print("Saved: {}".format(OUT_CLEAN_RESUME_PATH))

    if RAW_JD_PATH.exists():
        jobs = pd.read_csv(RAW_JD_PATH)
        if sample_size:
            jobs = jobs.head(sample_size).copy()
        clean_jobs = preprocess_job_descriptions(jobs, lemmatizer, stop_words)
        clean_jobs.to_csv(OUT_CLEAN_JD_PATH, index=False)
        print("Saved: {}".format(OUT_CLEAN_JD_PATH))
    else:
        print("No job descriptions file found at {} — skipping.".format(RAW_JD_PATH))

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
