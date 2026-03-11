# Phase 4.2: Ablation Studies

## Overview

Systematic ablation studies to evaluate the impact of design choices on code smell detection performance. All studies use the local **llama3:8b** model to ensure reproducibility and privacy-preservation without external API dependencies.

## Ablation Configurations

### 1. RAG vs No RAG (RQ2)

**Purpose**: Quantify the improvement from Retrieval-Augmented Generation

- **baseline_no_rag**: Pure LLM detection without knowledge base
- **rag_topk_1/3/5/10**: RAG with varying retrieval counts

**Expected Outcome**:
- Measure ΔF1 (improvement with RAG)
- Calculate false positive reduction rate
- Generate Figure 5: RAG top-k ablation study

**Metrics**:
- Precision@k, Recall@k per smell type
- NDCG@k for retrieval quality

### 2. Top-k Retrieval Values

**Purpose**: Find optimal balance between retrieval quantity and quality

Configurations: k ∈ {1, 3, 5, 10}

**Expected Outcome**:
- Identify optimal k value for M4 Pro context constraints
- 5-fold cross-validation on validation set
- Plot F1 vs k curve

### 3. Few-shot Examples

**Purpose**: Evaluate impact of in-context demonstration examples

Configurations: num_examples ∈ {0, 1, 3, 5}

**Expected Outcome**:
- Determine if few-shot improves over zero-shot with RAG
- Tradeoff between context usage and accuracy
- Figure 6: Few-shot sensitivity analysis

### 4. Temperature Settings

**Purpose**: Analyze exploration vs exploitation in LLM inference

Configurations: temperature ∈ {0.0, 0.1, 0.3}

**Expected Outcome**:
- Assess robustness across temperature settings
- Balance between reproducibility (low temp) and diversity (high temp)
- Figure 7: Temperature sensitivity analysis

### 5. RAG with MMR Reranking

**Purpose**: Improve diversity of retrieved examples

Configurations: MMR with diversity_lambda ∈ {0.5}

**Expected Outcome**:
- Compare standard retrieval vs Max Marginal Relevance
- Measure reduction in redundancy among retrieved samples

## Dataset & Evaluation

- **Knowledge Base**: Training set (60%) indexed in ChromaDB (fixed across all ablations)
- **Evaluation Set**: Test set (20% holdout) - never seen during tuning
- **Validation**: 5-fold cross-validation on validation set (20%)
- **Random Seed**: 42 (fixed for reproducibility)

## Running Ablation Studies

### Run all ablations:
```bash
python scripts/run_ablation_study.py --config exp/ablation_studies/config.json
```

### Run specific ablation:
```bash
python scripts/run_ablation_study.py --config exp/ablation_studies/config.json \
  --ablation rag_topk_3
```

### Save results with custom output directory:
```bash
python scripts/run_ablation_study.py --output-dir results/ablation_studies
```

## Expected Deliverables

### Tables
- `ablation_results.csv` - Summary of all ablations ranked by F1
- `topk_analysis.csv` - F1 vs k values plus statistical tests
- `fewshot_analysis.csv` - F1 vs num_examples
- `temperature_analysis.csv` - F1 vs temperature

### Figures
- **Figure 5**: RAG top-k ablation (bar chart: F1 vs k ∈ {1,3,5,10})
- **Figure 6**: Few-shot sensitivity (line: F1 vs num_examples)
- **Figure 7**: Temperature sensitivity (line: F1 vs temperature)
- **Ablation comparison**: All ablations ranked by F1 (bar chart)

### Data Files
- `ablation_results_{timestamp}.json` - Detailed results with configs
- `ablation_summary_{timestamp}.csv` - Quick lookup table
- `error_log_{timestamp}.csv` - Per-sample error breakdown

## Statistical Analysis

For each ablation comparison:
1. **Paired t-test**: H₀: μ_A = μ_B on F1 scores across samples
2. **Cohen's d**: Effect size interpretation (small/medium/large)
3. **Bootstrap CI**: 95% confidence intervals on mean F1
4. **Multiple comparisons**: Correct for multiple hypothesis tests if needed

## Expected Findings

Based on literature and project constraints:

| Ablation | Expected ΔF1 |  Confidence |
|----------|--------------|-------------|
| RAG vs No RAG | +8-12% | High (RQ2) |
| k=3 vs k=5 | -1-2% | Medium (diminishing returns) |
| Few-shot (1 ex) | +2-3% | Medium |
| Temp=0.1 vs 0.3 | 0-1% | Low (minimal variance) |
| MMR reranking | +1-2% | Low (small effect) |

## Model Context Constraints

**llama3:8b** has ~3K token context window:
- 1K tokens: Input code sample
- 1K tokens: Retrieved RAG examples (typically 3-5 samples)
- 1K tokens: Model output & buffer

Implications:
- Large code samples (>500 LOC) may be truncated
- Top-k retrieval limited by context (k ≤ 10 recommended)
- Few-shot examples compete with code for context space

## Next Steps (Integration with Phase 4.1)

1. Run ablations after Phase 3.3 experiments complete
2. Feed results into `scripts/analyze_results.py` for summary tables
3. Generate publication-ready figures for paper
4. Document findings in this README

## References

- WBS Phase 4.2 (Section 4.2)
- Benchmarking Strategy Section 5 (Cross-Validation)
- config.py: RAG_CONFIG, MODEL_PARAMS
- exp/ablation_studies/config.json: Ablation definitions
