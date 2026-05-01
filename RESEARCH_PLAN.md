# Research Plan — Code Smell Detection (Local LLM + RAG vs Static Tools)

> Source of truth: `idea.txt`, `docs/research/proposal-latex/proposal.tex`, `notebooks/RESEARCH_TODO.md`.

---

## 1. Research Questions (from proposal)

| RQ  | Question                                                                                       | Status |
| --- | ---------------------------------------------------------------------------------------------- | ------ |
| RQ1 | How accurately do **local LLMs** detect code smells vs. **static analysis tools**?             | Done (vanilla LLM + 5 baseline tools scored on test split). |
| RQ2 | Does **RAG** improve LLM detection over **vanilla prompting**?                                  | Blocked — no RAG predictions exist yet. |
| RQ3 | **Per-smell** and **per-language** strengths/weaknesses?                                       | Not started. |
| RQ4 | **Resource cost** (latency, memory, throughput) of local deployment?                           | Data exists in `resource_profile.json`; not yet aggregated. |

---

## 2. Dataset — `data/datasets/SmellyCodeDataset/`

- **Origin:** Honda Research Institute Europe — *Smelly Code Dataset* (MIT). Same domain example (pizza shop) re-implemented in 4 languages.
- **Languages & files (annotated, smelly):**
  - Java: 7 `.java` files
  - Python: 7 `.py` files
  - JavaScript: 7 `.js` files
  - C++: 13 `.cpp/.h` files
- **Total samples:** 28 annotated source files → **470 manual smell annotations** across 12+ smell types.
- **Splits already made** (`data/processed/`, seed=42, 60/20/20): train=16, val=6, test=6.
- **Each sample has:** `source_code`, `language`, `class_name`, `annotations[]` with `{smell_type, category, method, description, confidence}`, plus `loc`/`sloc`.
- **Smell taxonomy (5 categories):** Bloaters, Couplers, Change Preventers, Dispensables, Object-Orientation Abusers.
- **Bonus assets in dataset folder (use for free):**
  - `Analysis/sonarqube/sonarqube_findings.md` — SonarQube run + manual mapping of SQ rules → canonical smell categories. Skip re-running SQ on dataset, reuse this.
  - `Analysis/{ModelLevel,TypeLevel,...}/Cleaned_GPT4.0Detection.csv`, `Cleaned_DeePSeekDetection.csv`, `Cleaned_GroundTruth.csv` — pre-cleaned ground-truth + GPT-4 + DeepSeek predictions in CSV. **We can use these as extra reference baselines** without paying for API calls.
  - `Analysis/Cost.md` — token-cost analysis from the original authors.
  - `Prompts/`, `PlantUML/` — prompt templates and architecture diagrams the dataset authors used.

### ML-relevance insight

- Dataset is **tiny (28 files, 470 labels)** — too small to train from scratch. Use it strictly as an **evaluation benchmark**.
- It is **multi-label per file** (one file → many smells). Plan metrics as **micro/macro F1 over (file, smell\_type) pairs**, not single-label accuracy.
- It is **highly class-imbalanced** (Long Method / Large Class dominate; Refused Bequest / Parallel Inheritance are rare). Report per-smell counts next to per-smell F1 — small N = unreliable F1.
- **Same domain across 4 languages** = good for cross-language generalization tests, **bad** for "does it generalize to new domains" claims. Document this limitation.
- For **ML-based analysis**, the only fair use is:
  - LLM zero-/few-shot prompting (what we're doing).
  - RAG with `train + validation` as the retrieval corpus, `test` as the held-out set.
  - Optional: lightweight classical ML (TF-IDF / code metrics → logistic regression) as a *fourth* baseline to sanity-check that smells aren't trivially detectable.

---

## 3. What's already in `results/predictions/` for static tools

- **`baseline/`** — raw outputs already collected:
  - Java: `java_pmd.json`, `java_sonarqube.json`, `java_checkstyle.json`, `java_spotbugs.json`, `java_intellij.json`
  - Python: `python_pylint.json`, `python_flake8.json`
  - JavaScript: `javascript_eslint.json`
  - Plus extra timestamped SonarQube/PMD reruns.
- **`llm_vanilla/baseline_*`** — 6 vanilla LLM runs (codellama @ T=0.1, seed=42) with `metrics.json`, `results.jsonl`, `resource_profile.json`. Current best F1 = 0 in the latest run (parsing issues — must investigate).
- **`llm_rag/`** — **empty**. This is the main missing piece.
- **`tables/`** — empty in the new clean tree (RQ1 tables referenced in TODO are stale; need regeneration).

---

## 4. Experiments still needed to finish the paper

### Critical path (RQ2 — the core claim)
1. Build RAG knowledge base from `train + validation` only → persist in `chromadb_store/`.
2. Run RAG inference on `test.json` with **the exact same config** as `llm_vanilla/baseline_20260313_113259` (codellama, T=0.1, seed=42, default prompt). Output → `results/predictions/llm_rag/rag_<timestamp>/`.
3. Investigate vanilla-LLM F1=0 / JSON parse failures before claiming RAG "wins" — currently the comparison is unfair.
4. Re-run vanilla once parser is fixed, then RAG, so the matched pair is valid.

### Analysis & figures (RQ1 + RQ2 + RQ3)
5. Regenerate `results/tables/rq1_*` (currently empty) from `baseline/` JSONs + ground truth: per-tool P/R/F1, per-language F1, per-smell F1.
6. Build the canonical **smell-taxonomy mapping** (PMD/SQ/ESLint/Pylint rule → canonical 12-smell label). Reuse `Analysis/sonarqube/sonarqube_findings.md` mapping where possible.
7. Paired vanilla-vs-RAG: per-sample ΔF1, Wilcoxon signed-rank, McNemar on detected-vs-not.
8. Per-smell × system heatmap (rows = 12 smells, cols = {best static tool, vanilla, RAG}).
9. Per-language P/R/F1 table (Java, Python, JS — C++ optional since proposal scopes it out).

### Error analysis (RQ3)
10. Confusion table: predicted smell ≠ ground-truth smell on same method.
11. Tag 10 FP + 10 FN per system with: `wrong-smell-type`, `granularity-mismatch`, `missing-context`, `over-trigger-on-comment`, `language-syntax-failure`.
12. Spot-check 5 RAG retrievals per smell type → label `helpful` / `irrelevant` / `harmful`.

### Resource profiling (RQ4)
13. Aggregate `resource_profile.json` across all runs → table: model × {avg\_latency\_ms, p95\_latency\_ms, max\_memory\_mb, avg\_cpu\_pct, throughput}.
14. Local-LLM electricity cost vs. equivalent GPT-4/Claude API cost on the same token volume.

### Deliverable
15. Single notebook `notebooks/research_evaluation.ipynb` that loads predictions, regenerates every table/figure, dumps to `results/{tables,figures}/`, drop-in for the paper.

---

## 5. Short TODO (do these in order)

- [ ] Fix vanilla LLM JSON-parse failures (latest run F1=0 is a bug, not a result).
- [ ] Build RAG index from train+val, persist to `chromadb_store/`.
- [ ] Run RAG on test split — same model/seed as vanilla.
- [ ] Re-run vanilla after parser fix for matched comparison.
- [ ] Build smell-taxonomy mapping CSV (reuse SonarQube mapping in dataset).
- [ ] Regenerate `results/tables/rq1_combined.csv` (per-tool / per-language / per-smell P/R/F1).
- [ ] Compute paired stats (Wilcoxon, McNemar) for RAG vs vanilla.
- [ ] Per-smell × system F1 heatmap.
- [ ] Resource-profile table + cost comparison.
- [ ] Error-analysis tagging (10 FP + 10 FN per system).
- [ ] Wire everything into `notebooks/research_evaluation.ipynb`.
- [ ] Sync proposal scope: drop C++ claims if we don't run it; keep Java/Python/JS as in `proposal.tex`.

---

## 6. Out of scope (do not let these block the paper)

- Frontend, FastAPI, Docker, Alembic, LangGraph workflow, multi-agent orchestration — **all already removed**.
- New languages beyond Java/Python/JS (C++ optional).
- Training/fine-tuning custom models.
- Commercial-API evaluation (we already have GPT-4 + DeepSeek CSVs from the dataset authors if needed for reference).
- Real-time IDE integration.
