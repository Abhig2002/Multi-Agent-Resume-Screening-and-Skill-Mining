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
9. [Bias and Fairness (Course Requirement)](#9-bias-and-fairness-course-requirement)
10. [Project Structure](#10-project-structure)
11. [Implementation Plan](#11-implementation-plan)
12. [Work Division](#12-work-division)
13. [Setup & Installation](#13-setup--installation)
14. [References](#14-references)

---

## 1. Project Overview

Traditional resume screening relies on keyword matching and single-model scoring, which misses qualified candidates, fails to surface meaningful skill gaps, and provides limited interpretability. Single-model pipelines also struggle to handle the full range of tasks involved in screening — extracting skills, grouping candidates, finding skill patterns, classifying resumes, and matching to job descriptions are fundamentally different problems that benefit from different methods.

This project builds a **Multi-Agent Resume Screening & Skill Mining System** — a six-stage pipeline where each agent handles exactly one task. The novelty is the **multi-agent architecture**: separating skill extraction, clustering, association mining, classification, and job matching into independent components rather than forcing one model to do everything.

Two stages use an **LLM (Llama 3.1 8B Instruct)** where language understanding genuinely improves results over rule-based approaches. All other stages use classical data mining methods (K-Means, Apriori, SVM/Random Forest). The full system is evaluated against a plain TF-IDF single-model baseline.

### Why Only One LLM Stage?

| Stage | LLM? | Reason |
|---|---|---|
| Skill Extraction | ✅ Yes | spaCy misses domain-specific skills; LLM returns structured JSON |
| Clustering, ARM, Classification, Matching | ❌ No | These operate on structured data / embeddings — classical methods are faster, more reproducible, and easier to evaluate |

### LLM Backend — Two Options

The pipeline supports two interchangeable backends for the LLM stages, selected via `--llm-provider`:

| Backend | Flag | Model | Notes |
|---|---|---|---|
| **HuggingFace** (default) | `--llm-provider huggingface` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Runs locally, no rate limits, needs HF token + GPU recommended |
| **Groq API** | `--llm-provider groq` | `llama-3.1-8b-instant` | Cloud API, free tier (6K TPM), needs `GROQ_API_KEY` |

---

## 2. Problem Statement

The course project description frames **three core challenges** (see project brief). We align with them as follows:

1. **Structuring unstructured data**: Resumes are noisy and use varied vocabulary to describe the same skills. Rule-based NER misses domain-specific terms. We convert text into **structured skill JSON** (LLM) and **dense embeddings** (SBERT) for downstream mining.
2. **High-dimensional, sparse skill distributions**: Skill sets across candidates are wide and sparse — standard bag-of-words models struggle here. We use **SBERT**, **clustering**, and **association rules** to analyze structure without relying on a single sparse bag-of-words view alone.
3. **Bias in evaluation signals**: Automated screening can reflect **dataset imbalance**, **keyword priors**, or **uneven error across groups**. The public resume dataset does **not** include protected attributes (e.g. gender, race); we cannot measure demographic fairness directly. We **mitigate and report** what we can: **stratified train/val/test splits**, **class-weighted** classifiers where applicable, **per-class F1** and **disparity metrics** (spread of F1 across job categories), plus an explicit **limitations / ethics** discussion in the final report. **Interpretability** (association rules, LLM match reasons) complements scalar scores.

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
|           Skill Extraction Agent  [LLM]         |
|                                                  |
|  Option A (default): HuggingFace local           |
|    meta-llama/Meta-Llama-3.1-8B-Instruct         |
|  Option B: Groq API (llama-3.1-8b-instant)       |
|                                                  |
|  - Instruct chat template (system + user)        |
|  - Strict skill filtering + deduplication        |
|  - Output: {"technical": [...], "soft": [...]}   |
|  - Atomic JSON cache, saved every 10 rows        |
|  - tqdm progress bar; Ctrl-C saves and exits     |
|  - SBERT embeddings generated here              |
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
|          Job Matching Agent  [LLM]              |
|                                                  |
|  Step 1: SBERT cosine similarity -> top-10 list  |
|  Step 2: LLM reranks + explains fit             |
|  Output: {"score": 8, "reason": "..."}           |
|                                                  |
+--------------------------------------------------+

[LLM] = HuggingFace local (default) or Groq API -- switchable via --llm-provider
```

### Agent Summary

| Agent | Type | Powered By |
|---|---|---|
| Preprocessing Agent | Plain Python | NLTK, spaCy |
| **Skill Extraction Agent** | **LLM API call** | **Groq — Llama 3.1 8B / HuggingFace Llama 3.1** |
| Clustering Agent | Plain Python | scikit-learn K-Means + PCA |
| ARM Agent | Plain Python | mlxtend Apriori |
| Classification Agent | Plain Python | scikit-learn LinearSVC + RandomForest |
| Job Matching Agent | Plain Python | sentence-transformers cosine similarity |

---

## 6. Pipeline — Stage by Stage

### Stage 1 — Preprocessing Agent
**Type:** Plain Python
**Input:** Raw resume text and job description text
**What it does:** Strips HTML, lowercases, tokenizes, removes stopwords, lemmatizes
**Output:** `clean_resumes.csv` and `clean_jds.csv`

---

### Stage 2 — Skill Extraction Agent ⭐ LLM
**Type:** LLM — HuggingFace (local) or Groq API, switchable via `--llm-provider`
**Default model:** `meta-llama/Meta-Llama-3.1-8B-Instruct` (requires HF token + license acceptance)
**Input:** Clean resume text (`clean_resumes.csv`)
**What it does:**
- Uses instruct chat template (system + user messages) for structured JSON output
- Applies strict skill filtering: 1–5 word limit per skill, soft-skill denylist, proxy-term removal, deduplication, capped list sizes
- Atomic JSON cache written every 10 rows — crashes lose at most 10 entries
- `tqdm` progress bar shows speed, cached count, and per-resume skill counts
- Ctrl+C / SIGTERM handler saves cache before exiting — just re-run to continue
- SBERT embeddings (`all-MiniLM-L6-v2`) generated in the same stage

**Output:**
```json
{
  "technical": ["python", "sql", "kubernetes", "dbt"],
  "soft": ["leadership", "stakeholder management"]
}
```
Saved to `skill_lists.json` and `embeddings.npy`

**Why LLM here:** spaCy NER misses domain-specific terms like "dbt", "Kubernetes", "stakeholder management" that don't appear in standard named entity categories.

---

### Stage 3 — Clustering Agent
**Type:** Plain Python — K-Means (scikit-learn)
**Input:** `embeddings.npy`, `clean_resumes.csv`
**What it does:**
- L2-normalizes embeddings before clustering (cosine-consistent geometry; toggleable via `--no-normalize-embeddings`)
- Optional PCA dimensionality reduction before K-Means via `--pca-components N`
- Sweeps k in `[--k-min, --k-max]` (default 8–16), selecting best k by **silhouette score**
- Records **Davies-Bouldin** and **Calinski-Harabasz** scores per k as secondary metrics
- Post-hoc alignment metrics computed against `Category` labels (not used during clustering): **NMI** and **purity**
- Writes variant-specific summary files for ablation comparison (`clustering_summary_norm.json`, `clustering_summary_nonorm.json`, `clustering_summary_norm_pca64.json`)

**Ablation results (MiniLM 384-d, k∈[8,16]):**

| Variant | best\_k | Silhouette | Davies-Bouldin | Calinski-Harabasz | NMI | Purity |
|---------|---------|-----------|----------------|-------------------|-----|--------|
| Norm (no PCA) | 11 | 0.1008 | 2.588 | 97.1 | 0.439 | 0.400 |
| No norm | 11 | 0.1008 | 2.588 | 97.1 | 0.439 | 0.400 |
| Norm + PCA-64 | 10 | **0.1311** | **2.223** | **141.3** | 0.423 | 0.372 |

PCA-64 (retaining ~80% variance) improves all internal clustering metrics at a small cost to category alignment.

**Output:**
- `data/processed/cluster_assignments.csv` — per-resume cluster labels
- `evaluation/results/clustering_summary.json` — canonical latest
- `evaluation/results/stage3_results/` — per-variant summaries + ablation table CSV

---

### Stage 4 — Association Rule Mining Agent
**Type:** Plain Python — Apriori (mlxtend)
**Input:** `skill_lists.json`, `clean_resumes.csv`
**What it does:** Builds a binary skill transaction matrix from extracted technical + soft skills, runs Apriori with configurable `--min-support` (default 0.05) and `--min-threshold` confidence (default 0.5), sorts rules by lift.

**Results (min\_support=0.05, min\_confidence=0.5, full 2484-resume dataset):**

| Metric | Value |
|--------|-------|
| Unique skills | 12,164 |
| Rules mined | 330 |
| Lift range | 0.89 – 4.70 |
| Confidence range | 0.50 – 0.99 |

Top rules by lift:

| Antecedents | Consequents | Confidence | Lift |
|-------------|-------------|-----------|------|
| leadership, teamwork | time management, problem solving, communication | 0.574 | 4.70 |
| teamwork, problem solving | time management, leadership, communication | 0.529 | 4.49 |
| leadership, teamwork, communication | time management, problem solving | 0.621 | 4.41 |
| word | powerpoint, excel | 0.604 | 3.74 |
| word, excel | powerpoint | 0.631 | 3.51 |

Two dominant skill clusters emerge: a **soft skill cluster** (leadership / teamwork / communication / time management / problem solving) and an **MS Office cluster** (Word / Excel / PowerPoint). Both show strong co-occurrence well above random chance.

**Output:**
- `evaluation/results/stage4_results/association_rules.csv` — all mined rules
- `evaluation/results/stage4_results/frequent_itemsets.csv` — frequent skill itemsets
- `evaluation/results/stage4_results/skill_item_matrix_meta.json` — matrix metadata

---

### Stage 5 — Classification Agent
**Type:** Plain Python — scikit-learn
**Input:** `clean_resumes.csv`, `embeddings.npy`
**What it does:**
- **Baseline:** TF-IDF (max 30K features, bigrams, min_df=2) + LinearSVC (`class_weight='balanced'`)
- **Proposed:** SBERT embeddings + RandomForest (200 trees, `class_weight='balanced'`)
- Stratified 70/15/15 train/val/test split (fixed `random_state=42`)
- Evaluates both models on val and test splits; reports accuracy, macro-F1, weighted-F1, and disparity gap (max − min per-class F1)

**Results (MiniLM embeddings, full 2484-resume dataset):**

| Model | Split | Accuracy | Macro-F1 | Weighted-F1 |
|-------|-------|----------|----------|-------------|
| tfidf\_linearsvc (baseline) | val | 0.721 | 0.674 | 0.708 |
| tfidf\_linearsvc (baseline) | test | 0.716 | 0.666 | 0.706 |
| sbert\_rf (proposed) | val | 0.651 | 0.582 | 0.628 |
| sbert\_rf (proposed) | test | 0.627 | 0.561 | 0.607 |

TF-IDF + LinearSVC outperforms SBERT + RandomForest on this dataset. This is consistent with the literature on short-text classification where sparse high-precision features (TF-IDF) compete well against dense embeddings used with weaker classifiers.

**Output:**
- `evaluation/results/classification_comparison.csv`
- `evaluation/results/per_class_f1_disparity.csv`
- `evaluation/results/classification_test_predictions.csv`
- `evaluation/results/stage5_results/` — same files copied for stage-specific analysis

---

### Stage 6 — Job Matching Agent
**Type:** Plain Python — cosine similarity (sentence-transformers `all-MiniLM-L6-v2`)
**Input:** `clean_resumes.csv`, `clean_jds.csv`, `embeddings.npy`
**What it does:**
- Encodes job descriptions with a fresh MiniLM encoder
- Computes cosine similarity between each JD and all resume embeddings
- Returns top-K ranked resume IDs per job (configurable via `--match-top-k`, default 10)
- Computes heuristic **Precision@K** (K=5,10) by inferring job category from JD text and checking how many top-K resumes match that category

**Output:**
- `evaluation/results/matching_topk.csv` — ranked resume IDs per job
- `evaluation/results/matching_precision_at_k.csv` — mean P@5 and P@10
- `evaluation/results/stage6_results/` — same files for stage-specific analysis

**Note:** LLM reranking is not included in the current implementation. Cosine similarity alone is the matching mechanism; Precision@K is computed via a heuristic category-matching rule (not ground-truth relevance labels).

---

## 7. Technology Stack

| Component | Library / Tool | Notes |
|---|---|---|
| Text preprocessing | nltk, spaCy | en_core_web_sm model |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 |
| LLM local (default) | transformers, torch, accelerate | meta-llama/Meta-Llama-3.1-8B-Instruct; GPU recommended |
| LLM cloud (alternative) | groq | llama-3.1-8b-instant; free tier, 6K TPM |
| Clustering | scikit-learn (K-Means) | |
| Association rule mining | mlxtend | Apriori + FP-Growth |
| Classification | scikit-learn | SVM, Random Forest, Logistic Regression |
| Evaluation | scikit-learn + project helpers | accuracy, macro-F1, per-class F1, disparity, Precision@K |
| Progress + CLI | tqdm, argparse | Live progress bar, resumable runs |
| Caching | json + atomic os.replace | Crash-safe; saves every 10 rows |
| Environment vars | python-dotenv | GROQ_API_KEY, HF_TOKEN in .env |
| Version control | GitHub | |
| Communication | Discord | |

> **No fine-tuning required. GPU recommended for local HF inference but not mandatory.**

---

## 8. Evaluation Metrics

| Metric | Agent | Notes |
|---|---|---|
| Accuracy | Classification Agent | Overall correctness across 24 categories |
| Macro-F1 | Classification Agent | Primary metric — handles class imbalance fairly |
| Weighted-F1 | Classification Agent | Frequency-weighted F1 |
| Per-class F1 | Classification Agent | Identifies which categories are hardest |
| Disparity gap | Classification Agent | Max − min per-class F1 across categories (see §9) |
| Silhouette Score | Clustering Agent | Primary k-selection metric |
| Davies-Bouldin | Clustering Agent | Secondary metric; lower is better |
| Calinski-Harabasz | Clustering Agent | Secondary metric; higher is better |
| NMI / Purity | Clustering Agent | Post-hoc alignment with ground-truth Category labels |
| Cluster size stats | Clustering Agent | Min / max / mean cluster size |
| Support / Confidence / Lift | ARM Agent | Rule quality and interestingness |
| Precision@K (K=5, 10) | Job Matching Agent | Heuristic ranking quality |

**Primary comparison:** Multi-agent pipeline (SBERT + RF) vs. TF-IDF + LinearSVC baseline on classification
**Secondary comparisons:** PCA vs. no-PCA on clustering; normalization vs. no normalization on clustering

---

## 9. Bias and Fairness 

The original project description lists **detecting and mitigating bias in extracted evaluation signals** as a core challenge. Our approach:

- **Mitigation (engineering):** **Proxy term filter in Stage 2** strips demographic signals (university names, club affiliations, geographic identifiers) from extracted skill lists before they reach any downstream agent — implemented as `filter_proxy_skills()` in `skill_extraction_agent.py`. Stratified splits; class-weighted classification; transparent reporting of failures (confusion matrix, per-class F1). LLM outputs are treated as **assistive**; document temperature and caching behavior.
- **Analysis (reporting):** Report **per-class F1** and **disparity** (spread of F1 across the 24 job categories). A large gap between the best- and worst-served categories indicates **uneven reliability** of the system across labels — this is the main quantitative handle available without demographic fields.
- **Limitations:** We do **not** claim demographic fairness. Resume text can encode indirect cues; models may inherit dataset biases. Discuss limitations and cite relevant literature (e.g. README references on selection bias).

Implement disparity helpers alongside classification metrics in [`evaluation/metrics.py`](evaluation/metrics.py) (`compute_disparity_metrics`, `compute_classification_and_disparity`). Export tables to [`evaluation/results/`](evaluation/results/) for the final report.

---

## 10. Project Structure

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
│       ├── stage3_results/                      # Stage 3 — Clustering
│       │   ├── clustering_summary.json          # Latest run (default: norm)
│       │   ├── clustering_summary_norm.json     # L2-normalized K-Means
│       │   ├── clustering_summary_nonorm.json   # No normalization baseline
│       │   ├── clustering_summary_norm_pca64.json  # Norm + PCA-64 ablation
│       │   ├── clustering_ablation_table.csv    # Side-by-side ablation table
│       │   └── cluster_assignments.csv          # Per-resume cluster labels
│       ├── stage4_results/                      # Stage 4 — ARM
│       │   ├── association_rules.csv
│       │   ├── frequent_itemsets.csv
│       │   └── skill_item_matrix_meta.json
│       ├── stage5_results/                      # Stage 5 — Classification
│       │   ├── classification_comparison.csv
│       │   ├── per_class_f1_disparity.csv
│       │   └── classification_test_predictions.csv
│       └── stage6_results/                      # Stage 6 — Job Matching
│           ├── matching_topk.csv
│           └── matching_precision_at_k.csv
│
└── report/
    └── Group23_Final_Report.pdf
```

---

## 11. Implementation Plan

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
| Run full evaluation — accuracy, macro-F1, Precision@K, **per-class F1 + disparity** | Ishansh |
| Final report writing — include **bias/limitations** subsection (course challenge 3) | All |

---

## 12. Work Division

| Member | Responsibility |
|---|---|
| **Abhisekhar Bharadwaj Gandavarapu** | Preprocessing Agent (Stage 1) + Skill Extraction Agent (Stage 2) |
| **Tyler Tannenbaum** | Clustering Agent (Stage 3) + Integration notebook |
| **Ishansh Sharma** | Classification Agent (Stage 5) + evaluation metrics |
| **Ke Chen** | ARM Agent (Stage 4) + Job Matching Agent (Stage 6) |
| **All members** | Final report writing |

**Communication:** Discord · GitHub · Weekly in-class sync

---

## 13. Setup & Installation

### 1. Clone the repo and create a virtual environment
```bash
git clone <repo-url>
cd CSE572_Project
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install all dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure API keys
Copy the example env file and fill in your keys:
```bash
cp .env.example .env
```
Edit :
```
GROQ_API_KEY=gsk_...          # only needed for --llm-provider groq
HF_TOKEN=hf_...               # only needed for gated HF models (e.g. Llama 3.1)
```

- **Groq key**: [console.groq.com](https://console.groq.com) — free, no credit card
- **HF token**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — Read role

For Llama 3.1 you must also accept the license at:
[huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

### 4. Download datasets
Place the raw files in :
-  from [Kaggle — Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
-  from [Kaggle — Job Descriptions 2025](https://www.kaggle.com/datasets/adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles)

### 5. Run the pipeline

**Week 1 — preprocessing + skill extraction (Groq API, default):**
```bash
python run_pipeline.py --stage week1 --llm-provider groq
```

**Or use HuggingFace local inference:**
```bash
python run_pipeline.py --stage week1 --llm-provider huggingface
```

**If the run is interrupted, just re-run the same command — it resumes from the JSON cache automatically.**

**Regenerate embeddings only (skip LLM extraction):**
```bash
python run_pipeline.py --stage skills --skip-llm
```

**Week 2 — clustering, ARM, classification, matching (run all in sequence):**
```bash
python run_pipeline.py --stage week2
```

**Run individual stages:**
```bash
python run_pipeline.py --stage cluster
python run_pipeline.py --stage arm
python run_pipeline.py --stage classify
python run_pipeline.py --stage match
```

**Stage 3 — clustering with PCA dimensionality reduction:**
```bash
python run_pipeline.py --stage cluster --pca-components 64
```

**Stage 3 — disable L2 normalization (ablation):**
```bash
python run_pipeline.py --stage cluster --no-normalize-embeddings
```

**Stage 3 — widen k search range:**
```bash
python run_pipeline.py --stage cluster --k-min 6 --k-max 20
```

**Stage 4 — lower support threshold:**
```bash
python run_pipeline.py --stage arm --min-support 0.03
```

**Stage 6 — change number of top-K matches per job:**
```bash
python run_pipeline.py --stage match --match-top-k 5
```

**Quick smoke test on 50 resumes:**
```bash
python run_pipeline.py --stage week2 --sample-size 50
```

## 14. References

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

*Last updated: April 20, 2026 | CSE 572 Group 23 | Arizona State University*