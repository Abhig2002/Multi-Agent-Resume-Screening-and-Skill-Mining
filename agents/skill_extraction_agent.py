# stage 2: pull skills from resumes with groq or huggingface, then SBERT embeddings

import argparse
import json
import os
import re
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
GROQ_SLEEP = 10.0  # groq free tier is easy to hit
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
HF_MODEL_NAME = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
PROMPT_CHAR_LIMIT = 4000
CACHE_SAVE_EVERY = 10
MIN_TECH_FALLBACK = 3
MIN_SOFT_FALLBACK = 2

load_dotenv(ROOT / ".env")


def get_groq_client():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("need GROQ_API_KEY in env or .env")
    if Groq is None:
        raise ImportError("pip install groq")
    return Groq(api_key=key)


def load_hf_model(model_name: str):
    if not HAS_TRANSFORMERS:
        raise ImportError("need transformers and torch")

    hf_token = os.environ.get("HF_TOKEN") or None

    print("loading model", model_name)
    if hf_token:
        print("(using HF_TOKEN)")

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
    print("done, device =", device)
    return tokenizer, model


def _print_hf_403_help(err_text: str) -> None:
    low = err_text.lower()
    if "403" not in low and "forbidden" not in low and "gated" not in low:
        return
    print("got 403 from huggingface - check you accepted the model license + token can read gated repos")
    print("or try --hf-model-name Qwen/Qwen2.5-3B-Instruct")


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
        {"role": "user", "content": build_user_message(resume_text)},
    ]


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
    if not isinstance(technical, list):
        technical = []
    if not isinstance(soft, list):
        soft = []
    return {"technical": technical, "soft": soft}


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
    technical = sorted({_normalize(x) for x in skills.get("technical", []) if _keep_technical(x)})
    soft = sorted({_normalize(x) for x in skills.get("soft", []) if _keep_soft(x)})
    return {
        "technical": technical[:_MAX_TECHNICAL],
        "soft": soft[:_MAX_SOFT],
    }


def is_empty_skills(skills: Dict[str, List[str]]) -> bool:
    return not skills.get("technical") and not skills.get("soft")


def _fallback_extract_from_text(resume_text: str) -> Dict[str, List[str]]:
    # if llm returns nothing, scrape a few keywords so arm/clustering dont break
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


def load_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: Path, payload: Dict[str, Any]) -> None:
    # write temp file then rename so we dont corrupt json if killed mid-write
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


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
    device = next(model.parameters()).device

    def extract(resume_text: str) -> Dict[str, List[str]]:
        messages = build_chat_messages(resume_text)
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
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = parse_skill_json(text)
        return clean_skills(parsed)

    return extract


def generate_embeddings(texts: List[str], batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(SBERT_MODEL_NAME)
    all_batches = []
    batches = range(0, len(texts), batch_size)
    for i in tqdm(batches, desc="embeddings", unit="batch", dynamic_ncols=True):
        batch = texts[i : i + batch_size]
        all_batches.append(model.encode(batch, show_progress_bar=False))
    if not all_batches:
        dim = int(model.get_sentence_embedding_dimension())
        return np.empty((0, dim), dtype=np.float32)
    return np.vstack(all_batches)


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
    print(total, "resumes,", already_done, "cached,", empty_in_cache, "empty,", total - already_done, "left to run")

    if not skip_llm:
        provider = llm_provider.lower().strip()
        if provider == "groq":
            client = get_groq_client()
            extractor = _groq_extractor(client)
        elif provider == "huggingface":
            tokenizer, model = load_hf_model(hf_model_name)
            extractor = _hf_extractor(tokenizer, model, hf_max_new_tokens)
        else:
            raise ValueError("llm_provider should be groq or huggingface")

        print("using", provider)

        def _handle_signal(sig, frame):
            tqdm.write("saving and exiting...")
            save_cache(OUT_SKILLS_PATH, cache)
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        pending = [(str(row["ID"]), str(row.get("clean_text", "")))
                   for _, row in df.iterrows()
                   if (str(row["ID"]) not in cache)
                   or (refill_empty and is_empty_skills(cache.get(str(row["ID"]), {})))]

        processed = 0
        bar = tqdm(pending, total=len(pending), desc="skills", unit="r")
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
        print("cached", len(cache), "resumes,", recovered_empty, "needed fallback")
    else:
        print("skip llm")

    print("SBERT...")
    embeds = generate_embeddings(df["clean_text"].fillna("").astype(str).tolist())
    np.save(OUT_EMBED_PATH, embeds)
    print("wrote", OUT_EMBED_PATH, embeds.shape)
    print("stage 2 done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--llm-provider", choices=["groq", "huggingface"], default="groq")
    parser.add_argument("--hf-model-name", default=HF_MODEL_NAME)
    parser.add_argument("--hf-max-new-tokens", type=int, default=180)
    parser.add_argument("--refill-empty", action="store_true")
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
