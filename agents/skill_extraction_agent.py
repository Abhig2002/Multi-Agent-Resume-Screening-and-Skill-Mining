"""
Stage 2 - Skill Extraction Agent (Template)
===========================================
Boilerplate for:
1) LLM skill extraction with disk caching
2) Proxy/sensitive-term filtering for fairness mitigation
3) SBERT embeddings generation
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from groq import Groq
except ImportError:  # keeps sample workflow friendly even before pip install
    Groq = None


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
IN_CLEAN_RESUME_PATH = PROCESSED_DIR / "clean_resumes.csv"
OUT_SKILLS_PATH = PROCESSED_DIR / "skill_lists.json"
OUT_EMBED_PATH = PROCESSED_DIR / "embeddings.npy"

GROQ_MODEL = "llama-3.1-8b-instant"
RATE_LIMIT_SLEEP = 0.5
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


def get_client():
    """Initialize Groq client from environment variable."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("Missing GROQ_API_KEY in environment.")
    if Groq is None:
        raise ImportError("groq package not installed. Run: pip install groq")
    return Groq(api_key=key)


def build_extraction_prompt(resume_text: str) -> str:
    """Prompt that requests strict JSON output."""
    return f"""Extract all technical and soft skills from this resume.
Return ONLY valid JSON in this exact format:
{{"technical": ["skill1", "skill2"], "soft": ["skill3"]}}

Resume:
{resume_text[:2000]}"""


def parse_skill_json(raw: str) -> dict[str, list[str]]:
    """
    Parse model output safely.
    Falls back to empty lists on parse errors.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"technical": [], "soft": []}

    technical = data.get("technical", [])
    soft = data.get("soft", [])
    if not isinstance(technical, list):
        technical = []
    if not isinstance(soft, list):
        soft = []
    return {"technical": technical, "soft": soft}


def _normalize_skill(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def filter_proxy_skills(skills: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Remove non-skill / sensitive proxy-like terms.

    TODO (learning):
    - Expand this list based on error analysis.
    - Consider maintaining this list in a separate config file.
    """
    blocked_fragments = {
        "university",
        "college",
        "club",
        "association",
        "city",
        "state",
        "country",
    }

    def keep(item: str) -> bool:
        text = _normalize_skill(item)
        if not text:
            return False
        return not any(fragment in text for fragment in blocked_fragments)

    technical = sorted({_normalize_skill(x) for x in skills.get("technical", []) if keep(x)})
    soft = sorted({_normalize_skill(x) for x in skills.get("soft", []) if keep(x)})
    return {"technical": technical, "soft": soft}


def load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_skills_with_cache(
    client,
    resume_id: int | str,
    resume_text: str,
    cache: dict[str, Any],
) -> dict[str, list[str]]:
    """
    Extract once, cache forever.
    """
    key = str(resume_id)
    if key in cache:
        return cache[key]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": build_extraction_prompt(resume_text)}],
        temperature=0.1,
    )
    parsed = parse_skill_json(response.choices[0].message.content)
    filtered = filter_proxy_skills(parsed)
    cache[key] = filtered
    return filtered


def generate_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate SBERT embeddings for clean resume text.
    """
    model = SentenceTransformer(SBERT_MODEL_NAME)
    all_batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_batches.append(model.encode(batch, show_progress_bar=False))
    if not all_batches:
        return np.empty((0, 384), dtype=np.float32)
    return np.vstack(all_batches)


def run(sample_size: int | None = None, skip_llm: bool = False) -> None:
    """
    Stage 2 runner.
    - skip_llm=True lets you test embedding generation quickly without API calls.
    """
    df = pd.read_csv(IN_CLEAN_RESUME_PATH)
    if sample_size:
        df = df.head(sample_size).copy()

    cache = load_cache(OUT_SKILLS_PATH)
    if not skip_llm:
        client = get_client()
        for _, row in df.iterrows():
            rid = row["ID"]
            text = row["clean_text"]
            extract_skills_with_cache(client, rid, text, cache)
            save_cache(OUT_SKILLS_PATH, cache)
            time.sleep(RATE_LIMIT_SLEEP)
        print(f"Saved skills cache: {OUT_SKILLS_PATH}")
    else:
        print("Skipping LLM extraction (--skip-llm enabled).")

    embeds = generate_embeddings(df["clean_text"].fillna("").astype(str).tolist())
    np.save(OUT_EMBED_PATH, embeds)
    print(f"Saved embeddings: {OUT_EMBED_PATH} shape={embeds.shape}")
    print("Stage 2 complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 2 skill extraction + embeddings.")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip Groq calls and only generate embeddings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(sample_size=args.sample_size, skip_llm=args.skip_llm)
"""
Stage 2 — Skill Extraction Agent  [GROQ LLM + SBERT]
======================================================
Purpose:
    Extract structured skill lists from cleaned resume text using the
    Groq LLM API, then generate SBERT sentence embeddings for all resumes.

Why LLM here?
    spaCy Named Entity Recognition (NER) is trained on general text and
    misses domain-specific terms like "dbt", "Kubernetes", "Terraform",
    or "stakeholder management". An LLM understands context and returns
    clean JSON without needing a custom training dataset.

Why caching?
    The Groq free tier processes ~30 requests/minute. With 2,484 resumes,
    extraction takes ~90 minutes. Caching every result to disk means:
    - A Colab session crash loses nothing — just resume the loop.
    - Re-running experiments is instant (reads from disk, not API).

Input:
    data/processed/clean_resumes.csv   (output of Stage 1)

Output:
    data/processed/skill_lists.json    e.g. {"42": {"technical": [...], "soft": [...]}}
    data/processed/embeddings.npy      shape: (2484, 384)  — float32 SBERT vectors
"""

import json
import os
import time
import numpy as np
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration — set your Groq API key here or via environment variable
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
GROQ_MODEL = "llama-3.1-8b-instant"

# Rate limit: free Groq tier allows ~30 requests/min → sleep 0.5s between calls
RATE_LIMIT_SLEEP = 0.5

# SBERT model — downloads ~80MB on first use, cached locally after that
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Groq client (initialized once, reused across all calls)
# ---------------------------------------------------------------------------

client = Groq(api_key=GROQ_API_KEY)
sbert = SentenceTransformer(SBERT_MODEL_NAME)

# ---------------------------------------------------------------------------
# Skill Extraction
# ---------------------------------------------------------------------------

def build_extraction_prompt(resume_text: str) -> str:
    """
    Construct the prompt sent to the Groq LLM.

    Learning note — Prompt Engineering:
        - "Return ONLY valid JSON" suppresses the LLM from adding
          conversational text around the answer.
        - temperature=0.1 (near-zero) makes the output deterministic.
          Higher temperature = more creative but less reliable JSON.
        - We truncate to 2000 chars to stay within token limits and
          reduce latency. The most relevant skills appear early in resumes.

    Args:
        resume_text: Cleaned resume text (output of Stage 1).

    Returns:
        Formatted prompt string.
    """
    # TODO: Return a string with this exact structure:
    #
    #   f"""Extract all technical and soft skills from this resume.
    #   Return ONLY valid JSON with this exact format, nothing else:
    #   {{"technical": ["skill1", "skill2"], "soft": ["skill3"]}}
    #
    #   Resume:
    #   {resume_text[:2000]}"""
    #
    # Note: use double braces {{ }} to escape curly braces inside an f-string.
    raise NotImplementedError("Build and return the extraction prompt string")


def extract_skills(resume_text: str) -> dict:
    """
    Call the Groq API to extract skills from a single resume.

    Learning note — API call structure:
        client.chat.completions.create() follows the OpenAI chat format:
        - model: which LLM to use
        - messages: list of {"role": "user"/"assistant", "content": "..."}
        - temperature: 0.0 = fully deterministic, 1.0 = very random

    Args:
        resume_text: Raw or cleaned resume string.

    Returns:
        dict with keys "technical" and "soft", each a list of strings.
        Falls back to {"technical": [], "soft": []} on parse failure.
    """
    # TODO: Build the prompt using build_extraction_prompt(resume_text)
    #       Replace the line below with: prompt = build_extraction_prompt(resume_text)
    raise NotImplementedError("Call build_extraction_prompt here")

    # TODO: Call the Groq API:
    #   response = client.chat.completions.create(
    #       model=GROQ_MODEL,
    #       messages=[{"role": "user", "content": prompt}],
    #       temperature=0.1,
    #   )
    raise NotImplementedError("Make the Groq API call")

    # TODO: Extract the text content from the response:
    #   content = response.choices[0].message.content
    raise NotImplementedError("Extract response.choices[0].message.content")

    # TODO: Parse the JSON. Wrap in try/except json.JSONDecodeError so that
    #       a malformed response (LLM adds a sentence before the JSON) doesn't
    #       crash the whole extraction loop.
    #   try:
    #       return json.loads(content)
    #   except json.JSONDecodeError:
    #       return {"technical": [], "soft": []}
    raise NotImplementedError("Parse JSON with try/except fallback")


def extract_skills_cached(
    resume_id: str,
    resume_text: str,
    cache_path: str = "data/processed/skill_lists.json",
) -> dict:
    """
    Extract skills for a single resume, reading from and writing to a JSON cache.

    Learning note — Disk Caching Pattern:
        1. Load the entire cache dict from disk (or start empty).
        2. If resume_id is already in the cache, return it immediately.
        3. Otherwise, call the API, store result in cache, write back to disk.
        This is called a "write-through cache" — every new result is
        persisted immediately, so no work is lost on a crash.

    Args:
        resume_id:  Unique identifier (will be stored as a string key).
        resume_text: Cleaned resume text.
        cache_path:  Path to the JSON cache file.

    Returns:
        Skill dict: {"technical": [...], "soft": [...]}
    """
    # Step 1 — Load existing cache
    # TODO: If cache_path exists, read it with json.load().
    #       Otherwise start with cache = {}
    raise NotImplementedError("Load cache from disk or start empty dict")

    # Step 2 — Cache hit
    # TODO: if str(resume_id) in cache, return cache[str(resume_id)]
    raise NotImplementedError("Return cached result if it exists")

    # Step 3 — Cache miss: call the API
    # TODO: skills = extract_skills(resume_text)
    raise NotImplementedError("Call extract_skills for a cache miss")

    # Step 4 — Write result to cache and save to disk
    # TODO: cache[str(resume_id)] = skills
    #       Write cache back to cache_path using json.dump()
    raise NotImplementedError("Update cache dict and write to disk")

    # Step 5 — Rate limit
    # TODO: time.sleep(RATE_LIMIT_SLEEP)
    raise NotImplementedError("Sleep to respect Groq rate limits")

    # TODO: return skills
    raise NotImplementedError("Return the extracted skills dict")


def run_skill_extraction(
    clean_resumes_path: str = "data/processed/clean_resumes.csv",
    cache_path: str = "data/processed/skill_lists.json",
) -> dict:
    """
    Run skill extraction over all resumes, using the cache for efficiency.

    Args:
        clean_resumes_path: Path to clean_resumes.csv (Stage 1 output).
        cache_path:         Path to the JSON cache file.

    Returns:
        Dict mapping resume ID → {"technical": [...], "soft": [...]}
    """
    # TODO: Load clean_resumes.csv
    raise NotImplementedError("Load clean_resumes.csv with pd.read_csv")

    # TODO: Loop through rows and call extract_skills_cached() for each.
    #       Use df.iterrows() or df.itertuples().
    #       Print progress: f"[{i+1}/{len(df)}] Processing resume {row['ID']}"
    #
    #       Example:
    #   all_skills = {}
    #   for i, row in df.iterrows():
    #       skills = extract_skills_cached(row['ID'], row['clean_text'], cache_path)
    #       all_skills[str(row['ID'])] = skills
    #       if i % 10 == 0:
    #           print(f"[{i}/{len(df)}] ...")
    raise NotImplementedError("Loop through resumes and call extract_skills_cached")

    # TODO: return all_skills
    raise NotImplementedError("Return the full skills dict")


# ---------------------------------------------------------------------------
# SBERT Embeddings
# ---------------------------------------------------------------------------

def generate_embeddings(
    texts: list,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Generate SBERT sentence embeddings in batches.

    Learning note — Why batching?
        SBERT encodes text using a transformer model. Processing all 2,484
        resumes at once would exhaust RAM. Batching processes `batch_size`
        documents at a time and stacks the results.

    Learning note — What is an embedding?
        An embedding is a fixed-length vector (here: 384 floats) that captures
        the semantic meaning of text. Two resumes about Python/ML will have
        embeddings that are geometrically close (high cosine similarity),
        even if they use different words.

    Args:
        texts:      List of cleaned text strings.
        batch_size: Number of documents to encode at once.

    Returns:
        NumPy array of shape (len(texts), 384), dtype float32.
    """
    # TODO: Initialize an empty list: embeddings = []
    raise NotImplementedError("Initialize embeddings list")

    # TODO: Loop in steps of batch_size:
    #   for i in range(0, len(texts), batch_size):
    #       batch = texts[i : i + batch_size]
    #       batch_emb = sbert.encode(batch, show_progress_bar=False)
    #       embeddings.append(batch_emb)
    #       print(f"Encoded {min(i+batch_size, len(texts))}/{len(texts)} documents")
    raise NotImplementedError("Batch encode with sbert.encode and collect results")

    # TODO: Stack all batches into one array with np.vstack(embeddings)
    #       and return the result.
    raise NotImplementedError("Stack batches with np.vstack and return")


def save_embeddings(
    clean_resumes_path: str = "data/processed/clean_resumes.csv",
    output_path: str = "data/processed/embeddings.npy",
) -> np.ndarray:
    """
    Generate and save SBERT embeddings for all cleaned resumes.

    Args:
        clean_resumes_path: Path to clean_resumes.csv.
        output_path:        Where to save the .npy file.

    Returns:
        The embeddings array (also saved to disk).
    """
    # TODO: Load clean_resumes.csv and extract the 'clean_text' column as a list
    raise NotImplementedError("Load clean_resumes.csv and get texts list")

    # TODO: Call generate_embeddings(texts)
    raise NotImplementedError("Call generate_embeddings")

    # TODO: Save with np.save(output_path, embeddings)
    #       Print: f"Embeddings saved: shape={embeddings.shape}"
    raise NotImplementedError("Save embeddings with np.save and print confirmation")

    # TODO: return embeddings
    raise NotImplementedError("Return embeddings array")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    print("=== Stage 2: Skill Extraction Agent ===")
    print("\n[1/2] Extracting skills with Groq LLM (cached)...")
    skills = run_skill_extraction()
    print(f"      Done. {len(skills)} resumes processed.")

    print("\n[2/2] Generating SBERT embeddings...")
    emb = save_embeddings()
    print(f"      Done. Embeddings shape: {emb.shape}")

    print("\nStage 2 complete.")
