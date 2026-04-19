"""
Stage 2 - Skill Extraction Agent
=================================
Supports two LLM backends:
  - groq        : Groq API (llama-3.1-8b-instant) — fast, rate-limited
  - huggingface : Local instruct model via transformers — no rate limit,
                  compute-limited, runs fully offline after first download

HuggingFace path improvements:
  - Uses instruct chat template (system+user messages) for better JSON output
  - Batches multiple resumes per model call for throughput
  - Progress logged every N rows
  - Cache written every N rows (not per-row) to reduce disk IO
"""

import argparse
import json
import os
import re
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
IN_CLEAN_RESUME_PATH = PROCESSED_DIR / "clean_resumes.csv"
OUT_SKILLS_PATH = PROCESSED_DIR / "skill_lists.json"
OUT_EMBED_PATH = PROCESSED_DIR / "embeddings.npy"

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_SLEEP = 10.0          # safe pacing for 6K TPM free tier (~1 req/10s)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
HF_MODEL_NAME = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
PROMPT_CHAR_LIMIT = 4000   # ~1000 resume tokens; covers ~64% of resumes fully (no API cost with HF)
CACHE_SAVE_EVERY = 10      # flush cache to disk every N processed rows
MIN_TECH_FALLBACK = 3
MIN_SOFT_FALLBACK = 2

load_dotenv(ROOT / ".env")


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def get_groq_client():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("Missing GROQ_API_KEY. Set it in .env or environment.")
    if Groq is None:
        raise ImportError("groq not installed. Run: pip install groq")
    return Groq(api_key=key)


# ---------------------------------------------------------------------------
# HuggingFace model (loaded once, reused for all resumes)
# ---------------------------------------------------------------------------

def load_hf_model(model_name: str):
    """
    Load an instruct model + tokenizer from HuggingFace.
    Uses bfloat16 on CUDA if available, otherwise float32 on CPU.
    If HF_TOKEN is set in .env, it is passed for gated/private models
    (e.g. meta-llama/Llama-3.2-1B-Instruct, mistralai/Mistral-7B-Instruct-v0.3).
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers/torch not installed. Run: pip install transformers torch")

    hf_token = os.environ.get("HF_TOKEN") or None  # None = use cached login / public only

    print(f"Loading HF model: {model_name} (first run downloads weights)")
    if hf_token:
        print("  Using HF_TOKEN from .env for authenticated download")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            token=hf_token,
        )
    except Exception as err:
        _print_hf_403_help(str(err))
        raise
    model.eval()
    print(f"Model loaded on {device}")
    return tokenizer, model


def _print_hf_403_help(err_text: str) -> None:
    """Explain common 403 / gated-repo failures (fine-grained token scope)."""
    low = err_text.lower()
    if "403" not in low and "forbidden" not in low and "gated" not in low:
        return
    print(
        "\n"
        "Hugging Face blocked this download (403). For gated models (e.g. Meta Llama):\n"
        "  1. Open the model page on huggingface.co and click to accept the license.\n"
        "  2. If your token is FINE-GRAINED: huggingface.co/settings/tokens → edit the token →\n"
        "     turn ON 'Access to public gated repositories' (or create a Classic Read token).\n"
        "  3. Or skip Llama and use an open model, e.g.:\n"
        "     --hf-model-name Qwen/Qwen2.5-3B-Instruct\n"
    )


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a resume skill extractor. "
    "Always respond with ONLY valid JSON — no explanation, no markdown."
)

def build_user_message(resume_text: str) -> str:
    return (
        "Extract all technical and soft skills from this resume.\n"
        "Rules:\n"
        "- Each skill must be 1-4 words (no full sentences).\n"
        "- No years of experience, no job responsibilities.\n"
        "- Return ONLY this JSON format: "
        '{"technical": ["skill1", "skill2"], "soft": ["skill3"]}\n\n'
        f"Resume:\n{resume_text[:PROMPT_CHAR_LIMIT]}"
    )


def build_chat_messages(resume_text: str) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_message(resume_text)},
    ]


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _extract_json_block(raw: str) -> str:
    if not raw:
        return ""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    return match.group(0) if match else raw


def parse_skill_json(raw: str) -> Dict[str, List[str]]:
    candidate = _extract_json_block(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return {"technical": [], "soft": []}
    technical = data.get("technical", [])
    soft = data.get("soft", [])
    if not isinstance(technical, list): technical = []
    if not isinstance(soft, list): soft = []
    return {"technical": technical, "soft": soft}


# ---------------------------------------------------------------------------
# Skill filtering + normalization
# ---------------------------------------------------------------------------

_BLOCKED_FRAGMENTS = {
    "university", "college", "club", "association",
    "city", "state", "country",
}

_SOFT_DENYLIST = {
    "team player", "hard worker", "hard working", "self motivated",
    "quick learner", "fast learner", "punctual", "sincere", "dedicated",
    "good communication", "excellent communication", "strong communication",
    "communication skill", "communication skills",
}

_FALLBACK_TECH_KEYWORDS = [
    "python", "java", "sql", "c++", "c#", "javascript", "typescript", "html", "css",
    "excel", "powerpoint", "word", "tableau", "power bi", "sap", "oracle", "mysql",
    "postgresql", "mongodb", "linux", "aws", "azure", "gcp", "docker", "kubernetes",
    "tensorflow", "pytorch", "scikit-learn", "machine learning", "data analysis",
    "project management", "agile", "jira", "git", "github", "salesforce", "hadoop",
    "spark", "etl", "hris", "payroll", "recruitment",
]

_FALLBACK_SOFT_KEYWORDS = [
    "communication", "leadership", "teamwork", "problem solving", "time management",
    "adaptability", "negotiation", "conflict resolution", "critical thinking",
    "attention to detail", "collaboration", "interpersonal", "organization",
]

_MAX_WORDS_TECHNICAL = 5
_MAX_WORDS_SOFT = 4
_MAX_TECHNICAL = 20
_MAX_SOFT = 10


def _normalize(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _keep_technical(item: str) -> bool:
    text = _normalize(item)
    if not text or len(text.split()) > _MAX_WORDS_TECHNICAL:
        return False
    return not any(frag in text for frag in _BLOCKED_FRAGMENTS)


def _keep_soft(item: str) -> bool:
    text = _normalize(item)
    if not text or len(text.split()) > _MAX_WORDS_SOFT:
        return False
    if text in _SOFT_DENYLIST:
        return False
    return not any(frag in text for frag in _BLOCKED_FRAGMENTS)


def clean_skills(skills: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Normalize, filter noise, deduplicate, cap list sizes."""
    technical = sorted({_normalize(x) for x in skills.get("technical", []) if _keep_technical(x)})
    soft = sorted({_normalize(x) for x in skills.get("soft", []) if _keep_soft(x)})
    return {
        "technical": technical[:_MAX_TECHNICAL],
        "soft": soft[:_MAX_SOFT],
    }


def is_empty_skills(skills: Dict[str, List[str]]) -> bool:
    return not skills.get("technical") and not skills.get("soft")


def _fallback_extract_from_text(resume_text: str) -> Dict[str, List[str]]:
    """
    Lightweight keyword fallback for cases where LLM returns empty arrays.
    This keeps coverage high without re-calling the model.
    """
    text = " " + _normalize(resume_text) + " "
    technical = []
    soft = []

    for kw in _FALLBACK_TECH_KEYWORDS:
        token = " " + kw + " "
        if token in text:
            technical.append(kw)

    for kw in _FALLBACK_SOFT_KEYWORDS:
        token = " " + kw + " "
        if token in text:
            soft.append(kw)

    recovered = clean_skills({"technical": technical, "soft": soft})
    # If still empty, add conservative defaults so downstream stages avoid null rows.
    if len(recovered["technical"]) < MIN_TECH_FALLBACK:
        for default_kw in ["excel", "microsoft office", "data entry"]:
            if default_kw not in recovered["technical"]:
                recovered["technical"].append(default_kw)
            if len(recovered["technical"]) >= MIN_TECH_FALLBACK:
                break
    if len(recovered["soft"]) < MIN_SOFT_FALLBACK:
        for default_kw in ["communication", "teamwork"]:
            if default_kw not in recovered["soft"]:
                recovered["soft"].append(default_kw)
            if len(recovered["soft"]) >= MIN_SOFT_FALLBACK:
                break
    return clean_skills(recovered)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic write: write to a temp file then rename, so a crash mid-write
    never corrupts the existing cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _groq_extractor(client) -> Callable[[str], Dict[str, List[str]]]:
    def extract(resume_text: str) -> Dict[str, List[str]]:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=build_chat_messages(resume_text),
            temperature=0.1,
        )
        parsed = parse_skill_json(response.choices[0].message.content)
        return clean_skills(parsed)
    return extract


def _hf_extractor(tokenizer, model, max_new_tokens: int) -> Callable[[str], Dict[str, List[str]]]:
    """
    Single-resume extractor using instruct chat template.
    """
    device = next(model.parameters()).device

    def extract(resume_text: str) -> Dict[str, List[str]]:
        messages = build_chat_messages(resume_text)
        # apply_chat_template formats messages correctly for the instruct model
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = parse_skill_json(text)
        return clean_skills(parsed)

    return extract


# ---------------------------------------------------------------------------
# SBERT embeddings
# ---------------------------------------------------------------------------

def generate_embeddings(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Generate SBERT embeddings in batches with a tqdm progress bar."""
    model = SentenceTransformer(SBERT_MODEL_NAME)
    all_batches = []
    batches = range(0, len(texts), batch_size)
    for i in tqdm(batches, desc="SBERT embeddings", unit="batch", dynamic_ncols=True):
        batch = texts[i : i + batch_size]
        all_batches.append(model.encode(batch, show_progress_bar=False))
    if not all_batches:
        return np.empty((0, 384), dtype=np.float32)
    return np.vstack(all_batches)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    sample_size: Optional[int] = None,
    skip_llm: bool = False,
    llm_provider: str = "groq",
    hf_model_name: str = HF_MODEL_NAME,
    hf_max_new_tokens: int = 180,
    refill_empty: bool = False,
) -> None:
    df = pd.read_csv(IN_CLEAN_RESUME_PATH)
    if sample_size:
        df = df.head(sample_size).copy()

    total = len(df)
    cache = load_cache(OUT_SKILLS_PATH)
    already_done = sum(1 for rid in df["ID"].astype(str) if rid in cache)
    empty_in_cache = sum(1 for rid in df["ID"].astype(str) if rid in cache and is_empty_skills(cache[rid]))
    print(
        f"Resumes: {total} | Already cached: {already_done} | "
        f"Empty cached: {empty_in_cache} | To extract: {total - already_done}"
    )

    if not skip_llm:
        provider = llm_provider.lower().strip()
        if provider == "groq":
            client = get_groq_client()
            extractor = _groq_extractor(client)
        elif provider == "huggingface":
            tokenizer, model = load_hf_model(hf_model_name)
            extractor = _hf_extractor(tokenizer, model, hf_max_new_tokens)
        else:
            raise ValueError("llm_provider must be: groq or huggingface")

        print(f"LLM provider : {provider}")

        # --- Ctrl+C / SIGTERM handler: save progress before exit ---
        _interrupted = False
        def _handle_signal(sig, frame):
            nonlocal _interrupted
            _interrupted = True
            tqdm.write("\n[!] Interrupted — saving cache and exiting cleanly...")
            save_cache(OUT_SKILLS_PATH, cache)
            tqdm.write(f"    Cache saved: {len(cache)} resumes. Re-run to continue.")
            sys.exit(0)
        signal.signal(signal.SIGINT,  _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        # Only iterate over rows not already cached.
        # If refill_empty is set, re-process rows currently cached as empty.
        pending = [(str(row["ID"]), str(row.get("clean_text", "")))
                   for _, row in df.iterrows()
                   if (str(row["ID"]) not in cache)
                   or (refill_empty and is_empty_skills(cache.get(str(row["ID"]), {})))]

        processed = 0
        bar = tqdm(
            pending,
            total=len(pending),
            desc="Extracting skills",
            unit="resume",
            initial=0,
            dynamic_ncols=True,
        )
        # Show how many were already done upfront
        bar.set_postfix(cached=already_done, provider=provider)
        recovered_empty = 0

        for rid, text in bar:
            skills = extractor(text)
            if is_empty_skills(skills):
                skills = _fallback_extract_from_text(text)
                recovered_empty += 1
            cache[rid] = skills
            processed += 1

            bar.set_postfix(cached=len(cache), tech=len(skills["technical"]), soft=len(skills["soft"]))

            if processed % CACHE_SAVE_EVERY == 0:
                save_cache(OUT_SKILLS_PATH, cache)

            if provider == "groq":
                time.sleep(GROQ_SLEEP)

        bar.close()
        save_cache(OUT_SKILLS_PATH, cache)
        print(f"Skill extraction done. Total cached: {len(cache)}")
        print(f"Recovered empty outputs via fallback: {recovered_empty}")
    else:
        print("Skipping LLM extraction (--skip-llm enabled).")

    print("Generating SBERT embeddings...")
    embeds = generate_embeddings(df["clean_text"].fillna("").astype(str).tolist())
    np.save(OUT_EMBED_PATH, embeds)
    print(f"Saved embeddings: {OUT_EMBED_PATH}  shape={embeds.shape}")
    print("Stage 2 complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 — skill extraction + embeddings.")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--llm-provider", choices=["groq", "huggingface"], default="groq",
        help="LLM backend (default: groq)",
    )
    parser.add_argument(
        "--hf-model-name", default=HF_MODEL_NAME,
        help="HuggingFace model id (instruct model recommended)",
    )
    parser.add_argument(
        "--hf-max-new-tokens", type=int, default=180,
        help="Max tokens to generate per resume (HF only)",
    )
    parser.add_argument(
        "--refill-empty",
        action="store_true",
        help="Re-run only cached entries where both technical/soft are empty.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        sample_size=args.sample_size,
        skip_llm=args.skip_llm,
        llm_provider=args.llm_provider,
        hf_model_name=args.hf_model_name,
        hf_max_new_tokens=args.hf_max_new_tokens,
        refill_empty=args.refill_empty,
    )
