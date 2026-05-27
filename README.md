# CSE 572 — Group 23: Multi Agent Resume screening & Skill Mining pipeline

## What is this Project

We built a pipeline that cleans resumes, pulls out skills with an LLM, makes SBERT embeddings, then runs a bunch of separate steps: clustering, association rules, classification into job category, and matching resumes to job descriptions. Each step lives in its own file under `agents/` and `run_pipeline.py` runs them.

The only part that calls an LLM is **skill extraction** (Groq by default — you can switch to Hugging Face). Everything else is scikit-learn, mlxtend Apriori, sentence-transformers, etc.

## Quick start

1. Clone the repo, make a venv, `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add `GROQ_API_KEY` (and `HF_TOKEN` if you use Hugging Face)
3. Put Kaggle files in `data/raw/` as **`Resume.csv`** and **`job_dataset.csv`**
4. From the project root:

```bash
python run_pipeline.py --stage week1 --llm-provider groq   # preprocess + skills + embeddings
python run_pipeline.py --stage week2                     # cluster, ARM, classify, match
```

If skill extraction stops partway, run the same command again — it keeps a JSON cache. `--skip-llm` rebuilds embeddings only; `--refill-empty` retries rows where both skill lists were empty.

Other useful flags: `--stage cluster --pca-components 64`, `--stage arm --min-support 0.03`, `--match-top-k 5`

## Repo layout

- `run_pipeline.py` — main CLI
- `agents/` — one script per stage
- `data/raw/` — your Kaggle CSVs (often gitignored)
- `data/processed/` — cleaned CSVs, `skill_lists.json`, `embeddings.npy`
- `evaluation/results/stage*_results/` — CSV/JSON outputs per stage

We don’t use a `notebooks/` folder; everything goes through the Python scripts.

## Data

| | |
|---|---|
| Resumes | [Kaggle resume dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) — 2,484 rows, 24 `Category` labels |
| Job descriptions | [Kaggle JD 2025](https://www.kaggle.com/datasets/adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles) — used for matching (~1,068 rows after our cleaning) |

Stage 1 uses **NLTK** (stopwords + WordNet lemmatizer) only — no spaCy in preprocessing.

## Pipeline

1. **Preprocess** → `clean_resumes.csv`, `clean_jds.csv`
2. **Skills + SBERT** → `skill_lists.json`, `embeddings.npy` (MiniLM 384-d, same row order as `clean_resumes.csv`)
3. **Clustering** — K-Means on embeddings, pick k by silhouette; outputs in `evaluation/results/stage3_results/`
4. **ARM** — Apriori on skill lists; rules in `stage4_results/`
5. **Classification** — TF-IDF + LinearSVC vs SBERT + RandomForest vs SBERT + LinearSVC; stratified 70/15/15; results in `stage5_results/`
6. **Matching** — cosine similarity on SBERT (optional small category boost in the top pool); no LLM here. Outputs in `stage6_results/`

### Numbers we got (your run may differ)

**Classification — test split** (see `evaluation/results/stage5_results/classification_comparison.csv`):

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| TF-IDF + LinearSVC | 0.716 | 0.666 |
| SBERT + RandomForest | 0.681 | 0.620 |
| **SBERT + LinearSVC** | **0.764** | **0.724** |

**Clustering ablation** — see `evaluation/results/stage3_results/clustering_ablation_table.csv` and `clustering_summary*.json`. On our run, PCA-64 + L2 norm looked best on silhouette (e.g. k≈10).

**ARM:** ~12k unique skill strings, ~330 rules at default support 0.05 and confidence filter 0.5.

**Matching P@K** is a *heuristic*: “relevant” means the resume’s `Category` matched something we inferred from the JD text. Check `matching_precision_at_k.csv` for mean P@5/P@10 and the “nonempty relevant” columns — see `matching_agent.py` if you need the exact logic.

## Bias / fairness 

We don’t have demographic labels on this dataset. We used stratified splits, balanced weights where the classifier supports it, and we report macro-F1 plus per-class F1 / disparity across the 24 job categories. That’s not the same thing as proving fairness across people — we’re just being upfront about uneven performance across categories.


# Main Pipeline Components
- Preprocessing, skill extraction
- Clustering, wiring `run_pipeline.py` 
- Classification, metrics 
- ARM, matching 


## References 

1. Daryani et al., 2020 — NLP resume screening / similarity. *Topics in Intelligent Computing and Industry Design*.
2. Bevara et al., 2025 — Resume2Vec. *Electronics*.
3. Dsilva & Murari, 2025 — SkillSync. IJERT.
4. Mifta et al., 2023 — multi-stage LLM + summarization. ICSSIT.
5. Lo et al., 2025 — multi-agent LLM hiring. arXiv:2504.02870.
6. Sun et al., 2026 — CoMAI. arXiv:2603.16215.
7. Buyreu Real, 2025 — bias / invisible signals. UPF.
8. Castleman et al., 2026 — LLM resume validity. arXiv:2602.18550.
9. Devic, 2025 — Apple PhD fellowship statement.
10. Zhong et al., 2024 — AI and carbon. *Applied Economics*.
