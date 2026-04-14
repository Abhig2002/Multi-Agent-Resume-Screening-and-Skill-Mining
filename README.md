# Multi-Agent Resume Screening & Skill Mining System
**CSE 572: Data Mining — Group 23 Final Project**

> **Team Members:** Tyler Tannenbaum, Abhisekhar Bharadwaj Gandavarapu, Ishansh Sharma, Ke Chen
> **Course:** CSE 572 — Data Mining, Arizona State University

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
10. [Implementation Plan](#10-implementation-plan)
11. [Work Division](#11-work-division)
12. [Setup & Installation](#12-setup--installation)
13. [References](#13-references)

---

## 1. Project Overview

Traditional resume screening relies on keyword matching and single-model scoring, which misses qualified candidates, fails to surface meaningful skill gaps, and provides limited interpretability. Single-model pipelines also struggle to handle the full range of tasks involved in screening — extracting skills, grouping candidates, finding skill patterns, classifying resumes, and matching to job descriptions are fundamentally different problems that benefit from different methods.

This project builds a **Multi-Agent Resume Screening & Skill Mining System** — a six-stage pipeline where each agent handles exactly one task. The novelty is the **multi-agent architecture**: separating skill extraction, clustering, association mining, classification, and job matching into independent components rather than forcing one model to do everything.

Two stages use the **Groq LLM API (Llama 3.1 8B)** where language understanding genuinely improves results over rule-based approaches. All other stages use classical data mining methods (K-Means, Apriori, SVM/Random Forest). The full system is evaluated against a plain TF-IDF single-model baseline.

### Why Only Two LLM Stages?

| Stage | LLM? | Reason |
|---|---|---|
| Skill Extraction | ✅ Yes | spaCy misses domain-specific skills; LLM returns clean JSON |
| Job Matching | ✅ Yes | Cosine similarity shortlists candidates; LLM explains *why* they fit |
| Clustering, ARM, Classification | ❌ No | These operate on structured data — LLMs add no value here |

### Why Groq?

- Completely **free** tier (no credit card required)
- Fastest inference of any free LLM API
- Simple API, nearly identical to OpenAI's format
- Disk caching handles rate limits and Colab session crashes
- Llama 3.1 8B is sufficient for structured JSON extraction tasks

---

## 2. Problem Statement

Three core challenges motivate this project:

1. **Unstructured text → structured representation**: Resumes are noisy and use varied vocabulary to describe the same skills. Rule-based NER misses domain-specific terms.
2. **High-dimensional, sparse skill distributions**: Skill sets across candidates are wide and sparse — standard bag-of-words models struggle here.
3. **Limited interpretability**: Single-model pipelines produce a scalar score with no explanation of what drove it or what skills are missing.

---

## 3. Research Hypothesis

> **Main Question:** Does a multi-agent pipeline that separates skill extraction, clustering, association mining, classification, and job matching outperform a single TF-IDF baseline on resume screening tasks?

**Expected Results:**
- Groq LLM skill extraction will outperform spaCy NER on downstream classification F1
- SBERT embeddings will outperform TF-IDF on accuracy and macro-F1
- LLM reranking in the Job Matching Agent will improve Precision@K over cosine similarity alone
- Association rules will surface interpretable skill co-occurrence patterns not visible to a single model

---

## 4. Datasets

### Dataset 1 — Resume Dataset
| Property | Details |
|---|---|
| Source | [Kaggle — Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) |
| Size | 2,484 resumes |
| Features | `ID`, `Resume_str`, `Resume_html`, `Category` |
| Primary inputs | `Resume_str` (text), `Category` (label) |
| Classes | 24 job categories (IT, Finance, HR, Engineering, etc.) |
| Class balance | ~100–120 samples per category; a few minority classes below 100 |
| Split | 70% train / 15% validation / 15% test (stratified) |

### Dataset 2 — Job Descriptions
| Property | Details |
|---|---|
| Source | [Kaggle — Job Descriptions 2025](https://www.kaggle.com/datasets/adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles) |
| Size | 1,100 synthetic job descriptions across 55 roles |
| Features | `JobID`, `Title`, `ExperienceLevel`, `YearsOfExperience`, `Skills`, `Responsibilities`, `Keywords` |
| Primary inputs | `Title` + `Skills` + `Responsibilities` + `Keywords` (combined) |
| Used in | Stage 6 (Job Matching) only |

### Preprocessing Steps (Both Datasets)
- Remove HTML tags and formatting artifacts
- Convert to lowercase, tokenize, remove stopwords (NLTK)
- Lemmatize using spaCy `en_core_web_sm`
- Save to `clean_resumes.csv` and `clean_jds.csv`
- SBERT embeddings generated in Stage 2 and reused across all downstream agents

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
|  Step 1: SBERT cosine similarity -> top-10 list  |
|  Step 2: Groq LLM reranks + explains fit         |
|  Output: {"score": 8, "reason": "..."}           |
|                                                  |
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

---

## 6. Pipeline — Stage by Stage

### Stage 1 — Preprocessing Agent
**Type:** Plain Python
**Input:** Raw resume text and job description text
**What it does:** Strips HTML, lowercases, tokenizes, removes stopwords, lemmatizes
**Output:** `clean_resumes.csv` and `clean_jds.csv`

---

### Stage 2 — Skill Extraction Agent ⭐ LLM
**Type:** Groq API call (Llama 3.1 8B)
**Input:** Clean resume text
**What it does:**
- Sends each resume to Groq with a zero-temperature JSON prompt
- Returns structured skill lists per resume
- Caches results to disk after every call — Colab crashes don't lose progress
- Also generates SBERT embeddings (`all-MiniLM-L6-v2`) for all downstream agents

**Output:**
```json
{
  "technical": ["Python", "SQL", "Kubernetes", "dbt"],
  "soft": ["leadership", "stakeholder management"]
}
```
Saved to `skill_lists.json` and `embeddings.npy`

**Why LLM here:** spaCy NER misses domain-specific terms like "dbt", "Kubernetes", "stakeholder management" that don't appear in standard named entity categories.

---

### Stage 3 — Clustering Agent
**Type:** Plain Python — K-Means (scikit-learn)
**Input:** `embeddings.npy`
**What it does:** Groups resumes by skill similarity without using labels. Optimal k selected via elbow method and silhouette scores. Expected k between 8 and 12.
**Output:** Cluster assignments + cluster skill profiles

---

### Stage 4 — Association Rule Mining Agent
**Type:** Plain Python — Apriori / FP-Growth (mlxtend)
**Input:** `skill_lists.json` → binary skill matrix
**What it does:** Mines frequent skill co-occurrence patterns. Minimum support 0.05, minimum confidence 0.6. FP-Growth run in parallel for efficiency comparison.
**Output:** Rules like `{Python, SQL} → {Data Analyst}` with support, confidence, lift

---

### Stage 5 — Classification Agent
**Type:** Plain Python — scikit-learn
**Input:** `embeddings.npy` (agent) or TF-IDF features (baseline)
**What it does:**
- **Baseline:** TF-IDF → SVM or Logistic Regression
- **Agent version:** SBERT embeddings → RBF-kernel SVM and Random Forest

Both trained on 70% split, evaluated on 15% test split with class-weight balancing.
**Output:** Predicted job category per resume

---

### Stage 6 — Job Matching Agent ⭐ LLM
**Type:** Groq API call (Llama 3.1 8B)
**Input:** `embeddings.npy` + job description embeddings
**What it does:**
- Phase 1: SBERT cosine similarity → top-10 shortlist (zero API cost)
- Phase 2: Groq LLM called once per shortlisted candidate, returns score + explanation

**Output:**
```json
{"score": 8, "reason": "Strong Python and SQL match; lacks cloud experience."}
```

**Why LLM here:** Cosine similarity alone cannot explain *why* a candidate fits. LLM reranking adds interpretability.

---

## 7. Technology Stack

| Component | Library / Tool | Notes |
|---|---|---|
| Text preprocessing | `nltk`, `spaCy` | `en_core_web_sm` model |
| Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2` |
| LLM API | `groq` | Llama 3.1 8B Instant — free, no credit card |
| Clustering | `scikit-learn` (K-Means) | |
| Association rule mining | `mlxtend` | Apriori + FP-Growth |
| Classification | `scikit-learn` | SVM, Random Forest, Logistic Regression |
| Evaluation | `scikit-learn` metrics | accuracy, macro-F1, Precision@K |
| Caching | `json` + `os` | Saves LLM results between Colab sessions |
| Environment | Google Colab | Free tier sufficient |
| Version control | GitHub | |
| Communication | Discord | |

> **No fine-tuning. No local GPU. No paid APIs required.**

---

## 8. Evaluation Metrics

| Metric | Agent | Notes |
|---|---|---|
| Accuracy | Classification Agent | Overall correctness across 24 categories |
| Macro-F1 | Classification Agent | Primary metric — handles class imbalance fairly |
| Per-class F1 | Classification Agent | Identifies which categories are hardest |
| Silhouette Score | Clustering Agent | Cluster quality, used to select k |
| Support / Confidence / Lift | ARM Agent | Rule quality and interestingness |
| Precision@K (K=5, 10) | Job Matching Agent | Ranking quality |

**Primary comparison:** Multi-agent pipeline vs. TF-IDF + single classifier baseline
**Secondary comparison:** LLM skill extraction vs. spaCy NER on downstream classification F1

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
│       ├── clean_jds.csv                 # Output of Stage 1
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
│   └── 06_full_pipeline.ipynb            # Integration notebook
│
├── agents/
│   ├── preprocessing_agent.py
│   ├── skill_extraction_agent.py         # Groq API call + disk caching
│   ├── clustering_agent.py
│   ├── arm_agent.py
│   ├── classification_agent.py
│   └── matching_agent.py                 # Cosine shortlist + Groq reranking
│
├── evaluation/
│   ├── metrics.py
│   └── results/
│       ├── classification_report.csv
│       └── association_rules.csv
│
└── report/
    └── Group23_Final_Report.pdf
```

---

## 10. Implementation Plan

### Week 1 — Foundation
**Goal:** Preprocessing done, skill lists and embeddings generated. Everything else depends on this.

| Task | Owner |
|---|---|
| Download + explore both Kaggle datasets | Abhi + Tyler |
| Build preprocessing pipeline — Stage 1 | Abhi |
| Set up Groq API + skill extraction with caching — Stage 2 | Abhi |
| Generate + save SBERT embeddings | Abhi |

> ⚠️ **Critical:** Stages 3–6 all depend on `skill_lists.json` and `embeddings.npy`. Week 2 cannot start until these exist.

---

### Week 2 — Core Agents
**Goal:** All four core agents independently producing output.

| Task | Owner |
|---|---|
| Clustering Agent — K-Means on SBERT vectors | Tyler |
| ARM Agent — Apriori/FP-Growth on skill sets | Ke Chen |
| Classification Agent — TF-IDF baseline + SBERT version | Ishansh |
| Job Matching Agent — cosine shortlist + Groq reranking | Ke Chen |

---

### Week 3 — Integration & Report
**Goal:** Connected pipeline, full evaluation, final report submitted.

| Task | Owner |
|---|---|
| Connect all agents into `06_full_pipeline.ipynb` | Tyler |
| Run full evaluation — accuracy, macro-F1, Precision@K | Ishansh |
| Final report writing | All |

---

## 11. Work Division

| Member | Responsibility |
|---|---|
| **Abhisekhar Bharadwaj Gandavarapu** | Preprocessing Agent (Stage 1) + Skill Extraction Agent (Stage 2) |
| **Tyler Tannenbaum** | Clustering Agent (Stage 3) + Integration notebook |
| **Ishansh Sharma** | Classification Agent (Stage 5) + evaluation metrics |
| **Ke Chen** | ARM Agent (Stage 4) + Job Matching Agent (Stage 6) |
| **All members** | Final report writing |

**Communication:** Discord · GitHub · Weekly in-class sync

---

## 12. Setup & Installation

### Install All Dependencies
```python
# NLP + embeddings
!pip install spacy nltk sentence-transformers
!python -m spacy download en_core_web_sm

# LLM API (free)
!pip install groq

# Data mining
!pip install mlxtend

# Standard ML
!pip install scikit-learn pandas numpy matplotlib seaborn
```

### Groq API Setup
1. Go to [console.groq.com](https://console.groq.com) — free account, no credit card
2. Generate an API key
3. Store it:
```python
GROQ_API_KEY = "your_key_here"
```

---

### Skill Extraction Agent
```python
import json, os, time
from groq import Groq

client = Groq(api_key=GROQ_API_KEY)

def extract_skills(resume_text: str) -> dict:
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
        temperature=0.1
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"technical": [], "soft": []}


def extract_skills_cached(resume_id, resume_text,
                          cache_path="data/processed/skill_lists.json"):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    if str(resume_id) in cache:
        return cache[str(resume_id)]
    skills = extract_skills(resume_text)
    cache[str(resume_id)] = skills
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    time.sleep(0.5)
    return skills
```

---

### SBERT Embeddings
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

### Job Matching Agent
```python
from sklearn.metrics.pairwise import cosine_similarity

def get_shortlist(jd_embedding, resume_embeddings, k=10):
    scores = cosine_similarity([jd_embedding], resume_embeddings)[0]
    return np.argsort(scores)[::-1][:k]

def explain_match(resume_text: str, jd_text: str) -> dict:
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
    top_k_indices = get_shortlist(jd_embedding, resume_embeddings, k)
    results = []
    for idx in top_k_indices:
        match = explain_match(resumes[idx], jd_text)
        match["resume_id"] = int(idx)
        results.append(match)
        time.sleep(0.5)
    return sorted(results, key=lambda x: x["score"], reverse=True)
```

---

## 13. References

1. Daryani, C., et al. 2020. An automated resume screening system using NLP and similarity. *Topics in Intelligent Computing and Industry Design*, 2(2), 99–103.

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
