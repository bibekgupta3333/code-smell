# Research TODO — Smelly Code Empirical Evaluation

Tracks the **research-side** work (not app-building) needed to answer RQ1–RQ4
of the proposal. Tied to the Smelly Code Dataset under
`data/datasets/SmellyCodeDataset/` and the splits under `data/processed/`.

> **Core claim under test:** *Local LLM + RAG detects code smells more
> accurately and explainably than (a) traditional static-analysis tools and
> (b) vanilla local-LLM prompting — on intentionally-smelly multi-language
> code.*

---

## 0 · Repo state we already have

- [x] Smelly Code Dataset checked in (Java / Python / JS / C++, 28 files, 470 annotations)
- [x] Train/val/test splits at `data/processed/{train,validation,test}.json`
- [x] Baseline tool runs at `results/predictions/baseline/` (PMD, SonarQube, Checkstyle, SpotBugs, IntelliJ, ESLint, Pylint, Flake8)
- [x] Vanilla LLM runs at `results/predictions/llm_vanilla/baseline_*`
- [x] EDA notebook at `notebooks/eda_smelly_code_dataset.ipynb`
- [x] RAG pipeline code at `src/rag/` (vector store, retriever, embeddings)
- [ ] **No** RAG predictions yet — `results/predictions/llm_rag/` is empty

---

## 1 · RQ1 — Local LLM vs static-analysis tools

Goal: precision / recall / F1 of vanilla LLM vs each baseline tool on the
test split, normalized to a common smell taxonomy.

- [x] **1.1** Notebook section: load `data/processed/test.json` as ground truth (file → list of `{smell_type, category, method}`) — `notebooks/rq1_llm_vs_baselines.ipynb`
- [x] **1.2** Normalize baseline tool outputs (`results/predictions/baseline/*.json`) into a single long-form DataFrame: `{file, language, tool, predicted_smell, line, confidence}` → `results/tables/rq1_baseline_predictions_long.csv`
- [x] **1.3** Build a smell-taxonomy mapping table (PMD/SQ rule → canonical smell from `CODE_SMELL_TYPES`); rules outside the catalogue tagged `Other` → `results/tables/rq1_taxonomy_mapping.csv`, `rq1_unmapped_rules.csv`
- [x] **1.4** Aggregate per-file predictions per tool; file-level multi-label P/R/F1 vs ground truth
- [x] **1.5** Same metrics for vanilla LLM runs in `results/predictions/llm_vanilla/baseline_*/results.jsonl`
- [x] **1.6** Combined RQ1 table: tool, P, R, F1, micro-F1, macro-F1, per-language F1 → `results/tables/rq1_combined.{csv,md}`

---

## 2 · RQ2 — Does RAG help?

Goal: side-by-side P/R/F1 of vanilla LLM vs RAG-LLM with **identical** model,
prompt, and seed.

- [ ] **2.1** Build / verify the RAG knowledge base from `train` + `validation` splits only (never test) → `chromadb_store/`
- [ ] **2.2** Run RAG pipeline on the test split using `scripts/experiments/run_experiment.py --type rag` (same model + temp + seed as the vanilla run we'll compare against)
- [ ] **2.3** Notebook section: load matched vanilla vs RAG runs (same model, same prompt variant)
- [ ] **2.4** Paired per-sample F1 plot (already prototyped in `results/notebook_benchmarks/per_sample_f1_rag_vs_plain.png`) — redo on the **test** split
- [ ] **2.5** Significance: Wilcoxon signed-rank on per-sample F1, also a McNemar test on per-(file, smell) detected-vs-not labels
- [ ] **2.6** Report: Δprecision, Δrecall, ΔF1, Δfalse-positive rate (RAG − vanilla)

---

## 3 · RQ3 — Per-smell-type / per-language breakdown

Goal: which of the 12 smell types and which languages are easy vs hard.

- [ ] **3.1** Per-smell P/R/F1 table for: (a) best static tool, (b) vanilla LLM, (c) RAG-LLM
- [ ] **3.2** Per-language P/R/F1 table for the same three systems
- [ ] **3.3** Heatmap: rows = smell type, cols = system, value = F1
- [ ] **3.4** Note structural smells (Long Method, Large Class, Long Parameter List) vs semantic smells (Feature Envy, Shotgun Surgery, Divergent Change) — does RAG help more on semantic ones?

---

## 4 · Error analysis

Goal: qualitative + quantitative explanation of *why* models miss / over-call.

- [ ] **4.1** Build a confusion-style table of mis-predictions (predicted smell ≠ ground-truth smell on the same method/class)
- [ ] **4.2** Sample 10 false positives and 10 false negatives per system, eyeball the code, tag each with one of: `wrong-smell-type`, `granularity-mismatch`, `missing-context`, `over-trigger-on-comment`, `language-syntax-failure` (note: vanilla runs show many `Syntax error: Syntax error at line 1: invalid syntax` rows we should investigate — see `results/notebook_benchmarks/runs_*_no_rag.csv`)
- [ ] **4.3** Did RAG-retrieved examples actually look relevant? Spot-check 5 retrievals per smell type, classify as `helpful` / `irrelevant` / `harmful`

---

## 5 · RQ4 — Deployment feasibility

Goal: latency, memory, throughput numbers that say whether this is realistic
on a developer laptop / CI.

- [ ] **5.1** Pull `resource_profile.json` from each `llm_vanilla/baseline_*` and the new RAG runs
- [ ] **5.2** Table: model × {avg_latency_ms, p95_latency_ms, max_memory_mb, avg_cpu_pct, throughput_files_per_min}
- [ ] **5.3** Compare local-LLM cost ($0 + electricity) to a same-throughput cloud-LLM call estimate (GPT-4 / Claude API list price × tokens used)
- [ ] **5.4** Note: 7B vs 13B vs quantized — which point on the size/F1 curve we recommend

---

## 6 · Notebook deliverable

Single research notebook at `notebooks/research_evaluation.ipynb` that:

- [ ] **6.1** Loads ground truth + all three system outputs from `results/predictions/`
- [ ] **6.2** Produces every table/figure listed above
- [ ] **6.3** Saves figures to `results/figures/` and tables to `results/tables/` so the paper can pick them up

---

## 7 · Out of scope (do not block research on these)

- Frontend / FastAPI integration (`FRONTEND_*.md`, `FASTAPI_IMPLEMENTATION.md`)
- New tool integrations beyond the 5 baselines listed in RQ1
- New models beyond `llama3:8b`, `codellama`, one Mistral variant
