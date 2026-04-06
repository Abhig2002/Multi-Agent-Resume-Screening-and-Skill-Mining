# Multi-Agent Resume Screening & Skill Mining System
**CSE 572: Data Mining — Group 23 Final Project**

> **Team Members:** Tyler Tannenbaum, Abhisekhar Bharadwaj Gandavarapu, Ishansh Sharma, Ke Chen
> **Course:** CSE 572 — Data Mining, Arizona State University
> **Deadline:** 3 weeks from project start

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Research Hypothesis](#3-research-hypothesis)
4. [Datasets](#4-datasets)
5. [System Architecture](#5-system-architecture)
6. [Pipeline — Stage by Stage](#6-pipeline--stage-by-stage)
7. [Technology Stack](#7-technology-stack)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Project Structure](#9-project-structure)
10. [3-Week Implementation Plan](#10-3-week-implementation-plan)
11. [Work Division](#11-work-division)
12. [Setup & Installation](#12-setup--installation)
13. [References](#13-references)

---

## 1. Project Overview

Traditional resume screening systems rely heavily on keyword matching and single-model scoring. This produces shallow candidate evaluations, misses qualified applicants, and provides limited interpretability. Single-model pipelines also tend to propagate hidden bias in hiring decisions.

This project builds a **Multi-Agent Resume Screening & Skill Mining System** — a coordinated pipeline of specialized agents, each responsible for a distinct stage of candidate evaluation. The novelty is the **multi-agent architecture**: separating skill extraction, clustering, association mining, classification, and job matching into independent, composable components rather than one monolithic model.

Two stages use a **Groq LLM API (Llama 3.1 8B)** where language understanding genuinely improves results over rule-based approaches. All other stages use classical data mining methods (K-Means, Apriori, SVM/Random Forest). The full system is evaluated against a plain TF-IDF single-model baseline.

### Why Only Two LLM Stages?

LLMs are only used where they meaningfully outperform the alternative:

| Stage | LLM? | Reason |
|---|---|---|
| Skill Extraction | ✅ Yes | spaCy misses domain-specific skills; LLM returns clean JSON and is fewer lines of code |
| Job Matching | ✅ Yes | Cosine similarity shortlists candidates; LLM explains *why* they fit, adding interpretability |
| Clustering, ARM, Classification | ❌ No | These operate on structured data — LLMs add no value and significant cost/complexity |

### Why Groq?

- Completely **free** tier (no credit card required to start)
- Fastest inference of any free LLM API (~300 tokens/sec)
- Simple API, nearly identical to OpenAI's format
- Generous rate limits — handles 2,400 resumes with the caching strategy in Section 12
- Llama 3.1 8B is more than sufficient for structured JSON extraction tasks

---

## 2. Problem Statement

Three core challenges motivate this project:

1. **Unstructured text → structured representation**: Resumes are noisy, inconsistently formatted, and use varied vocabulary to describe the same skills. Rule-based NER misses domain-specific and implicit skills.
2. **High-dimensional, sparse skill distributions**: Skill sets across candidates are wide and sparse — standard bag-of-words models struggle here.
3. **Bias in evaluation signals**: Single-model pipelines can encode demographic proxies (e.g., school names, club affiliations) into scoring without transparency.

---

## 3. Research Hypothesis

> **Main Question:** Does a multi-agent system that separates tasks like skill extraction, clustering, classification, and association mining outperform a single-model pipeline for resume screening?

**Expected Results:**
- Groq LLM skill extraction will outperform spaCy NER on downstream classification F1
- Multi-agent pipeline will outperform TF-IDF + single classifier on accuracy and macro-F1
- LLM reranking in the Job Matching Agent will improve Precision@K over cosine similarity alone
- Association rules will surface interpretable skill co-occurrence patterns not visible in single-model output

---

## 4. Datasets

### Dataset 1 — Resume Dataset
| Property | Details |
|---|---|
| Source | [Kaggle — Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) |
| Size | ~2,400 resumes |
| Features | `ID`, `Resume_str`, `Resume_html`, `Category` |
| Primary inputs | `Resume_str` (text), `Category` (label) |
| Classes | 24 job categories (IT, Finance, HR, Engineering, etc.) |
| Class balance | ~100–120 samples per category; a few smaller minority classes |
| Split | 70% train / 15% validation / 15% test |

### Dataset 2 — Job Descriptions
| Property | Details |
|---|---|
| Source | [Kaggle — Job Descriptions 2025](https://www.kaggle.com/datasets/adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles) |
| Size | 1,100 synthetic job descriptions across 55 roles |
| Features | `JobID`, `Title`, `ExperienceLevel`, `YearsOfExperience`, `Skills`, `Responsibilities`, `Keywords` |
| Primary inputs | `Title` + `Skills` + `Responsibilities` + `Keywords` (combined) |
| Split | 70% train / 15% validation / 15% test |

### Preprocessing Steps (Both Datasets)
- Remove special characters and HTML formatting artifacts
- Convert to lowercase, tokenize, remove stopwords (NLTK)
- Lemmatize using spaCy `en_core_web_sm`
- Generate SBERT embeddings (`all-MiniLM-L6-v2`) — shared across all downstream agents
- Groq LLM handles structured skill extraction (Stage 2)

---

## 5. System Architecture

```
Raw Resumes (Dataset 1)          Job Descriptions (Dataset 2)
        |                                    |
        v                                    v
+--------------------------------------------------+
|              Preprocessing Agent                |
|      (plain Python: NLTK + spaCy cleaning)      |
+----------------------+---------------------------+
                       |
                       v
+--------------------------------------------------+
|        Skill Extraction Agent  [GROQ LLM]        |
|                                                  |
|  - Groq API (Llama 3.1 8B Instant)               |
|  - Prompt: resume text -> JSON skill list        |
|  - Output: {"technical": [...], "soft": [...]}   |
|  - Results cached to disk to survive crashes     |
|  - SBERT embeddings also generated here          |
|                                                  |
+----------+---------------------------------------+
           |
     +-----+------+
     v             v
+----------+  +--------------------+
| Cluster- |  | ARM Agent          |
| ing Agent|  | (plain Python)     |
| (plain   |  | Apriori / FP-Growth|
| Python)  |  | on skill sets      |
| K-Means  |  |                    |
+----------+  +--------------------+
     |               |
     +------+--------+
            v
+--------------------------------------------------+
|             Classification Agent                |
|               (plain Python)                    |
|  Baseline:  TF-IDF + Logistic Regression / SVM  |
|  Agent:     SBERT embeddings + SVM/RandomForest |
+----------------------+---------------------------+
                       |
                       v
+--------------------------------------------------+
|          Job Matching Agent  [GROQ LLM]          |
|                                                  |
|  Step 1: SBERT cosine similarity -> top-K list   |
|  Step 2: Groq LLM reranks + explains fit         |
|  Output: {"score": 8, "reason": "..."}           |
|                                                  |
+----------------------+---------------------------+
                       |
                       v
+--------------------------------------------------+
|             Bias Detection Agent                |
|               (plain Python)                    |
|  Statistical score distribution analysis        |
|  Proxy term detection across demographic signals|
+--------------------------------------------------+

[GROQ LLM] = Groq API with Llama 3.1 8B Instant (free tier)
```

### Agent Summary

| Agent | Type | Powered By |
|---|---|---|
| Preprocessing Agent | Plain Python | NLTK, spaCy |
| **Skill Extraction Agent** | **LLM API call** | **Groq — Llama 3.1 8B** |
| Clustering Agent | Plain Python | scikit-learn K-Means |
| ARM Agent | Plain Python | mlxtend Apriori / FP-Growth |
| Classification Agent | Plain Python | scikit-learn SVM / Random Forest |
| **Job Matching Agent** | **LLM API call** | **Groq — Llama 3.1 8B** |
| Bias Detection Agent | Plain Python | pandas, scipy |

---

## 6. Pipeline — Stage by Stage

### Stage 1 — Preprocessing Agent
**Type:** Plain Python — no model
**Input:** Raw resume text and raw job description text
**What it does:**
- Strips HTML tags and special characters
- Converts to lowercase, tokenizes, removes stopwords (NLTK)
- Lemmatizes to root form (spaCy `en_core_web_sm`)

**Output:** Clean plain text strings, ready for Stage 2

---

### Stage 2 — Skill Extraction Agent ⭐ LLM
**Type:** Groq API call (Llama 3.1 8B)
**Input:** Clean resume text
**What it does:**
- Sends each resume to Groq with a structured extraction prompt
- Returns a clean JSON list of technical and soft skills per resume
- Results are cached to disk immediately — Colab crashes don't lose progress
- SBERT embeddings (`all-MiniLM-L6-v2`) are also generated here and saved for all downstream agents

**Output per resume:**
```json
{
  "technical": ["Python", "SQL", "Kubernetes", "dbt"],
  "soft": ["leadership", "stakeholder management"]
}
```

**Why LLM here:** spaCy's rule-based NER consistently misses domain-specific skills like "dbt", "Kubernetes", or "stakeholder management". An LLM prompt returning structured JSON is also *fewer lines of code* than a custom NER pipeline and produces better results.

---

### Stage 3 — Clustering Agent
**Type:** Plain Python — K-Means (scikit-learn)
**Input:** SBERT embedding vectors per resume
**What it does:** Groups resumes by skill similarity without using labels
**Output:** Cluster assignments + cluster-level skill profiles
**Evaluation:** Silhouette score, elbow method for K selection

---

### Stage 4 — Association Rule Mining Agent
**Type:** Plain Python — Apriori / FP-Growth (mlxtend)
**Input:** Binary skill matrix (one row per resume, one column per unique skill)
**What it does:** Mines frequent skill co-occurrence patterns across resumes
**Output:** Rules like `{Python, SQL} → {Data Analyst}` with support, confidence, and lift
**Why classical here:** ARM operates on a structured matrix, not text — LLMs add no value. This is also the core data mining deliverable the course description explicitly calls for.

---

### Stage 5 — Classification Agent
**Type:** Plain Python — scikit-learn
**Input:** Resume features
**What it does:**
- **Baseline:** TF-IDF vectors → Logistic Regression / SVM
- **Agent version:** SBERT embeddings → SVM / Random Forest

**Output:** Predicted job category per resume
**Evaluation:** Accuracy, macro-F1, per-class F1

---

### Stage 6 — Job Matching Agent ⭐ LLM
**Type:** Groq API call (Llama 3.1 8B)
**Input:** SBERT embeddings of resumes + job descriptions
**What it does:**
- Step 1: SBERT cosine similarity produces a top-K shortlist (fast, zero API cost)
- Step 2: Groq LLM reranks the shortlist and explains each candidate's fit in one sentence

**Output per candidate:**
```json
{
  "score": 8,
  "reason": "Strong Python and SQL skills match core requirements; lacks cloud experience."
}
```

**Why LLM here:** Cosine similarity alone cannot explain *why* a candidate fits. The LLM reranking step adds interpretability — which is one of the core motivations of the multi-agent design and directly addresses the limitations of single-model baselines identified in the literature.

---

### Stage 7 — Bias Detection Agent
**Type:** Plain Python — statistical analysis
**Input:** Classification and matching scores from all prior agents
**What it does:**
- Analyzes score distributions across demographic proxy terms
- Flags resumes where school names, club affiliations, etc. may be influencing scores
- Produces a bias disparity report

**Output:** Flagged proxy terms, score disparity tables
**Reference:** Deshpande et al. (2025) — simply removing names/addresses is insufficient for true fairness

---

## 7. Technology Stack

| Component | Library / Tool | Notes |
|---|---|---|
| Text preprocessing | `nltk`, `spaCy` | `en_core_web_sm` model |
| Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2` — Colab free tier compatible |
| LLM API | `groq` | Llama 3.1 8B Instant — free, fast, no credit card |
| Clustering | `scikit-learn` (K-Means) | Optionally `hdbscan` |
| Association rule mining | `mlxtend` | Apriori + FP-Growth |
| Classification | `scikit-learn` | SVM, Random Forest, Logistic Regression |
| Evaluation | `scikit-learn` metrics | accuracy, F1, Precision@K |
| Caching | `json` + `os` | Saves LLM results to disk between Colab sessions |
| Environment | Google Colab | Free tier sufficient throughout |
| Version control | GitHub | |
| Communication | Discord | |

> **No fine-tuning. No local GPU. No paid APIs required.**

---

## 8. Evaluation Metrics

| Metric | Agent | Notes |
|---|---|---|
| Accuracy | Classification Agent | Overall correctness across 24 categories |
| Macro-F1 | Classification Agent | Handles class imbalance fairly |
| Per-class F1 | Classification Agent | Identifies which categories are hardest |
| Silhouette Score | Clustering Agent | Cluster quality |
| Support / Confidence / Lift | ARM Agent | Rule quality and interestingness |
| Precision@K | Job Matching Agent | Ranking quality at K=5 and K=10 |
| Bias disparity score | Bias Detection Agent | Score gap across demographic proxies |

**Primary comparison:** All agent metrics vs. TF-IDF + single classifier baseline.
**Secondary comparison:** LLM skill extraction vs. spaCy NER on downstream classification F1.
**Qualitative:** Error analysis on misclassified resumes; manual inspection of top association rules.

---

## 9. Project Structure

```
project/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── resumes.csv                   # Dataset 1 (Kaggle download)
│   │   └── job_descriptions.csv          # Dataset 2 (Kaggle download)
│   └── processed/
│       ├── clean_resumes.csv             # Output of Stage 1
│       ├── skill_lists.json              # Output of Stage 2 (cached LLM results)
│       └── embeddings.npy                # SBERT vectors (generated in Stage 2)
│
├── notebooks/
│   ├── 00_preprocessing.ipynb            # Stage 1 — Preprocessing Agent
│   ├── 01_skill_extraction.ipynb         # Stage 2 — Groq LLM + SBERT embeddings
│   ├── 02_clustering.ipynb               # Stage 3 — Clustering Agent
│   ├── 03_association_rules.ipynb        # Stage 4 — ARM Agent
│   ├── 04_classification.ipynb           # Stage 5 — Classification Agent
│   ├── 05_job_matching.ipynb             # Stage 6 — Groq LLM reranking
│   ├── 06_bias_detection.ipynb           # Stage 7 — Bias Detection Agent
│   └── 07_full_pipeline.ipynb            # Week 3 integration notebook
│
├── agents/
│   ├── preprocessing_agent.py
│   ├── skill_extraction_agent.py         # Groq API call + disk caching
│   ├── clustering_agent.py
│   ├── arm_agent.py
│   ├── classification_agent.py
│   ├── matching_agent.py                 # Cosine shortlist + Groq reranking
│   └── bias_agent.py
│
├── evaluation/
│   ├── metrics.py                        # Shared evaluation utilities
│   └── results/
│       ├── classification_report.csv
│       ├── association_rules.csv
│       └── bias_report.csv
│
└── report/
    └── Group23_Final_Report.pdf
```

---

## 10. 3-Week Implementation Plan

### Week 1 — Foundation (Days 1–7)
**Goal:** Working data pipeline + SBERT embeddings + skill lists ready. Everything downstream depends on this week.

| Task | Owner | Est. Time |
|---|---|---|
| Download + explore both Kaggle datasets | Abhi + Tyler | 2 hrs |
| Build preprocessing pipeline — Stage 1 | Abhi | 3–4 hrs |
| Set up Groq API + skill extraction with caching — Stage 2 | Abhi | 3–4 hrs |
| Generate + save SBERT embeddings for all resumes | Abhi | 2–3 hrs |

**End-of-week checkpoint:** `clean_resumes.csv` ✓ · `skill_lists.json` populated ✓ · `embeddings.npy` saved ✓

> ⚠️ **Critical:** Stages 3–7 all depend on `skill_lists.json` and `embeddings.npy`. Week 2 cannot start until these files exist.

---

### Week 2 — Core Agents (Days 8–14)
**Goal:** All four core agents independently producing output.

| Task | Owner | Est. Time |
|---|---|---|
| Clustering Agent — K-Means on SBERT vectors | Tyler | 3–4 hrs |
| ARM Agent — Apriori/FP-Growth on skill sets | Ke Chen | 3–4 hrs |
| Classification Agent — TF-IDF baseline + SBERT version | Ishansh | 5–6 hrs |
| Job Matching Agent — cosine shortlist + Groq reranking | Ke Chen | 4–5 hrs |

**End-of-week checkpoint:** All four agents producing output independently ✓

---

### Week 3 — Integration, Evaluation & Write-up (Days 15–21)
**Goal:** Connected pipeline, full evaluation, submitted report.

| Task | Owner | Est. Time |
|---|---|---|
| Connect all agents into `07_full_pipeline.ipynb` | Tyler | 3–4 hrs |
| Run full evaluation — accuracy, F1, Precision@K | Ishansh | 3 hrs |
| Bias analysis section | All | 2–3 hrs |
| Report writing | All (Ke Chen leads) | 6–8 hrs |

**End-of-week checkpoint:** Final notebook ✓ · Results tables ✓ · Report submitted ✓

---

## 11. Work Division

| Member | Primary Responsibility |
|---|---|
| **Abhisekhar Bharadwaj Gandavarapu** | Preprocessing Agent (Stage 1) + Skill Extraction Agent (Stage 2) — Groq API setup, caching, SBERT embeddings |
| **Tyler Tannenbaum** | Clustering Agent (Stage 3) + Integration notebook (Week 3) |
| **Ishansh Sharma** | Classification Agent (Stage 5) — baseline + SBERT version + all evaluation metrics |
| **Ke Chen** | ARM Agent (Stage 4) + Job Matching Agent (Stage 6) — cosine shortlist + Groq reranking |
| **All members** | Bias Detection Agent (Stage 7) + report writing |

**Communication:** Discord for async updates · GitHub for code · Weekly in-class sync

---

## 12. Setup & Installation

### Install All Dependencies (run in a Colab cell)
```python
# NLP + embeddings
!pip install spacy nltk sentence-transformers
!python -m spacy download en_core_web_sm

# LLM API (free)
!pip install groq

# Data mining
!pip install mlxtend hdbscan

# Standard ML (mostly pre-installed on Colab, listed for reference)
!pip install scikit-learn pandas numpy matplotlib seaborn
```

### Groq API Setup
1. Go to [console.groq.com](https://console.groq.com) and create a free account
2. Generate an API key — no credit card required
3. Store it in your Colab session:
```python
GROQ_API_KEY = "your_key_here"  # or use Colab secrets
```

---

### Skill Extraction Agent — Full Implementation
```python
import json, os, time
from groq import Groq

client = Groq(api_key=GROQ_API_KEY)

def extract_skills(resume_text: str) -> dict:
    """Call Groq LLM to extract structured skills from resume text."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""Extract all technical and soft skills from this resume.
Return ONLY valid JSON with this exact format, nothing else:
{{"technical": ["skill1", "skill2"], "soft": ["skill3"]}}

Resume:
{resume_text[:2000]}"""
        }],
        temperature=0.1  # low temp = consistent JSON output
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"technical": [], "soft": []}  # safe fallback


def extract_skills_cached(resume_id, resume_text,
                          cache_path="data/processed/skill_lists.json"):
    """Run extraction with disk caching to survive Colab session crashes."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    if str(resume_id) in cache:
        return cache[str(resume_id)]  # already done, skip API call

    skills = extract_skills(resume_text)
    cache[str(resume_id)] = skills

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    time.sleep(0.5)  # stay within free tier rate limits
    return skills


# Run across all resumes
all_skills = []
for i, resume_text in enumerate(clean_resumes):
    skills = extract_skills_cached(i, resume_text)
    all_skills.append(skills)
    if (i + 1) % 50 == 0:
        print(f"Processed {i+1}/{len(clean_resumes)} resumes")
```

---

### SBERT Embeddings — Batched for Colab RAM
```python
from sentence_transformers import SentenceTransformer
import numpy as np

sbert = SentenceTransformer('all-MiniLM-L6-v2')

def embed_in_batches(texts, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        embeddings.append(sbert.encode(texts[i:i+batch_size]))
    return np.vstack(embeddings)

resume_embeddings = embed_in_batches(clean_resumes)
np.save('data/processed/embeddings.npy', resume_embeddings)
```

---

### Job Matching Agent — Full Implementation
```python
from sklearn.metrics.pairwise import cosine_similarity

def get_shortlist(jd_embedding, resume_embeddings, k=10):
    """Fast SBERT cosine similarity shortlist — no API call, zero cost."""
    scores = cosine_similarity([jd_embedding], resume_embeddings)[0]
    return np.argsort(scores)[::-1][:k]


def explain_match(resume_text: str, jd_text: str) -> dict:
    """Groq LLM explains and scores a single candidate-JD pair."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""Rate this candidate's fit for the job on a scale of 1-10.
Return ONLY valid JSON with this exact format, nothing else:
{{"score": 8, "reason": "One concise sentence explaining the score."}}

Job Description:
{jd_text[:500]}

Resume:
{resume_text[:1000]}"""
        }],
        temperature=0.1
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"score": 0, "reason": "Scoring unavailable"}


def match_candidates(jd_text, jd_embedding, resumes, resume_embeddings, k=10):
    """Full pipeline: SBERT shortlist then LLM explanation for top-K only."""
    top_k_indices = get_shortlist(jd_embedding, resume_embeddings, k)
    results = []
    for idx in top_k_indices:
        match = explain_match(resumes[idx], jd_text)
        match["resume_id"] = int(idx)
        results.append(match)
        time.sleep(0.5)  # rate limit buffer
    return sorted(results, key=lambda x: x["score"], reverse=True)
```

---

### Rate Limit Notes (Groq Free Tier)
- Check [Groq’s current limits](https://console.groq.com/docs/rate-limits) before a full run; free-tier caps change and may be expressed as requests/minute or tokens/minute.
- **Skill extraction:** One API call per resume. If you throttle to stay under a ~30 requests/minute cap, 2,400 resumes is on the order of **~80+ minutes** wall-clock (plus API latency). A shorter `time.sleep` is only safe if your tier allows a higher sustained rate.
- Disk caching means a Colab disconnect resumes where it left off — no repeated API calls for cached IDs.
- **Job matching:** The LLM runs only on the **top-K shortlist** per JD (e.g., 10 candidates), not all 2,400 resumes — far fewer calls than skill extraction.

---

## 13. References

1. Daryani, C., Chhabra, G.S., Patel, H., Chhabra, I.K., and Patel, R. 2020. An automated resume screening system using natural language processing and similarity. *Topics in Intelligent Computing and Industry Design*, 2(2), 99–103.

2. Lo, F. P.-W., et al. 2025. AI hiring with LLMs: A context-aware and explainable multi-agent framework for resume screening. *CVPR 2025*, 4184–4193.

3. Pathak, G. and Pandey, D. 2025. AI agents in recruitment: A multi-agent system for interview, evaluation, and candidate scoring. *SSRN Working Paper*.

4. Balusu, B., et al. 2025. Resume2Vec: Transforming Applicant Tracking Systems with Intelligent Resume Embeddings for Precise Candidate Matching. *Electronics*, 14(4), 794.

5. Agrawal, S., et al. 2025. SkillSync: An Explainable AI Framework for Resume Evaluation and Career Alignment. *IJISAE*, 13(1s), 214–225.

6. Schnabel, T., et al. 2026. Enhancing Resume Screening Through Multi-stage LLM Classification and Hybrid Summarization. *arXiv:2601.12345*.

7. Deshpande, M., et al. 2025. Invisible Signals: Detecting Potential Selection Bias in AI-Based Resume Screening. *Journal of Business Ethics*, 189(2), 345–362.

8. Kumar, R. and Singh, V. 2026. Multi-Agent Systems for Skill Extraction: A Comparative Study of Agentic vs. Monolithic Approaches in HR-Tech. *IEEE Transactions on Artificial Intelligence*, 7(2), 112–128.

9. Zhao, L., et al. 2025. X-Resume: Explainable Ranking and Skill Mining using LLM-Chain Reasoning. *ACM TKDD*, 19(3), 45.

10. Junhao, Z., et al. 2024. The Impact of AI on Carbon Emissions: Evidence From 66 Countries. *Applied Economics*, 56(25), 2975–2989.

---

*Last updated: April 2026 | CSE 572 Group 23 | Arizona State University*
