# Scripts Organization

## Organized by Functional Groups & Project Phases

```
scripts/
├── data/              # Phase 3.1-3.2: Data processing & indexing
├── experiments/       # Phase 3.3, 4.2: Experiment runners & ablations
├── analysis/          # Phase 4.1: Result analysis & evaluation
└── baseline/          # Baseline tool execution (PMD, SonarQube, etc)
```

---

## 📁 Directory Structure & Usage

### **scripts/data/** — Data Pipeline (Phase 3.1-3.2)

Handles dataset loading, preprocessing, indexing, and validation.

#### Files:

| Script | Purpose | Phase | Command |
|--------|---------|-------|---------|
| **`index_datasets.py`** | Index preprocessed data into ChromaDB | 3.2 | `python scripts/data/index_datasets.py` |
| **`test_retrieval_quality.py`** | Evaluate RAG retrieval quality on validation/test set | 3.2 | `python scripts/data/test_retrieval_quality.py --top-k 3` |
| **`validate_config.py`** | Validate project configuration (databases, paths, models) | Setup | `python scripts/data/validate_config.py` |

#### Key Features:
- ✅ Loads processed JSON splits (train/validation/test from Phase 3.1)
- ✅ Outputs ChromaDB collections in `chromadb_store/`
- ✅ Evaluates retrieval hit rate per smell type
- ✅ Validates system configuration before experiments

---

### **scripts/experiments/** — Experiment Runners (Phase 3.3, 4.2)

Executes code smell detection experiments with configurable parameters.

#### Files:

| Script | Purpose | Phase | Command |
|--------|---------|-------|---------|
| **`run_experiment.py`** | Main experiment executor (baseline LLM & RAG) | 3.3 | `python scripts/experiments/run_experiment.py --dataset test` |
| **`run_ablation_study.py`** | Systematic ablation study runner | 4.2 | `python scripts/experiments/run_ablation_study.py --config exp/ablation_studies/config.json` |

#### Key Features:
- ✅ **run_experiment.py**:
  - Loads test set via `--dataset test` or files via `--input`
  - Runs ExperimentExecutor with different configurations
  - Saves predictions + metrics to `results/predictions/{system}/`
  - Tracks resource usage (CPU, memory, latency)
  - Integrates with db for result tracking

- ✅ **run_ablation_study.py**:
  - Configuration-driven from `exp/ablation_studies/config.json`
  - Runs individual ablation via `--ablation {name}`
  - Runs all 13 ablations sequentially
  - Saves results as CSV + JSON summaries

---

### **scripts/analysis/** — Result Analysis (Phase 4.1)

Aggregates experiment results, generates benchmark tables, and creates visualizations.

#### Files:

| Script | Purpose | Phase | Command |
|--------|---------|-------|---------|
| **`analyze_results.py`** | Aggregate results from multiple runs; generate Tables 1-3 | 4.1 | `python scripts/analysis/analyze_results.py` |

#### Key Features:
- ✅ Loads metrics from `results/metrics/` or `results/predictions/`
- ✅ Aggregates across multiple runs (mean ± std)
- ✅ **Generates Benchmark Tables**:
  - Table 1: Overall Performance Comparison (6+ systems)
  - Table 2: Per-Smell-Type Performance (12+ smell types)
  - Table 3: Resource Requirements (latency, memory, CPU)
- ✅ Exports as CSV + LaTeX for paper
- ✅ Produces ranked system summaries

---

### **scripts/baseline/** — Baseline Tool Execution

Runs baseline code smell detection tools (PMD, SonarQube, Checkstyle, SpotBugs).

#### Key Scripts:
- `run_tools.py` — Execute all baseline tools on a sample
- `sonarqube_run.sh` — Setup & run SonarQube
- `normalize_output.py` — Convert tool outputs to standard JSON
- `install_tools.sh` — Install baseline tool dependencies
- `run_tools_docker.sh` — Run tools in Docker containers

---

## 🔄 Typical Workflow

### **Setup Phase**
```bash
# Validate configuration
python scripts/data/validate_config.py

# Install baseline tools (optional, for comparison)
bash scripts/baseline/install_tools.sh
```

### **Data Preparation (Phase 3.1-3.2)**
```bash
# Index training set into ChromaDB
python scripts/data/index_datasets.py --input-dir data/processed

# Test retrieval quality
python scripts/data/test_retrieval_quality.py --split test --top-k 3
```

### **Experiments (Phase 3.3)**
```bash
# Run baseline LLM (no RAG)
python scripts/experiments/run_experiment.py --input exp/baseline/config.json

# Run RAG-enhanced LLM
python scripts/experiments/run_experiment.py --input exp/rag_experiments/config.json
```

### **Ablation Studies (Phase 4.2)**
```bash
# Run all 13 ablations
python scripts/experiments/run_ablation_study.py --config exp/ablation_studies/config.json

# Run specific ablation
python scripts/experiments/run_ablation_study.py --ablation rag_topk_5
```

### **Analysis (Phase 4.1)**
```bash
# Aggregate results from all runs
python scripts/analysis/analyze_results.py --output-dir results/analysis_results

# Generate benchmark tables for paper
# (outputs Table1.csv, Table2.csv, Table3.csv + LaTeX versions)
```

---

## 📊 Output Locations

| Script | Output | Location |
|--------|--------|----------|
| `data/index_datasets.py` | ChromaDB collections | `chromadb_store/` |
| `data/test_retrieval_quality.py` | Retrieval metrics | `results/metrics/rag_retrieval_quality.json` |
| `experiments/run_experiment.py` | Predictions + metrics | `results/predictions/{system}/` |
| `experiments/run_ablation_study.py` | Ablation results | `results/ablation_studies/` |
| `analysis/analyze_results.py` | Benchmark tables | `results/analysis_results/` |
| `baseline/*.py` | Tool predictions | `results/predictions/baseline/` |

---

## 🎯 Command Reference

```bash
# Data Pipeline
python scripts/data/validate_config.py
python scripts/data/index_datasets.py [--clear] [--batch-size 32]
python scripts/data/test_retrieval_quality.py [--top-k 5] [--split test]

# Experiments
python scripts/experiments/run_experiment.py [--dataset test|train|validation]
python scripts/experiments/run_experiment.py [--input path/to/config.json]

# Ablations (all from one script)
python scripts/experiments/run_ablation_study.py
python scripts/experiments/run_ablation_study.py --ablation rag_topk_3

# Analysis
python scripts/analysis/analyze_results.py [--output-dir results/analysis_results]

# Baseline Tools
python scripts/baseline/run_tools.py [--sample-dir samples/]
bash scripts/baseline/install_tools.sh
bash scripts/baseline/sonarqube_run.sh
```

---

## 🔗 Related Modules

All scripts import from core project modules:

```python
from config import CODE_SMELL_TYPES, DATA_DIR, RESULTS_DIR, RAG_CONFIG
from src.data.data_loader import CodeSample
from src.data.data_preprocessor import DataPreprocessor
from src.rag.vector_store import VectorStore
from src.workflow.analysis_coordinator import AnalysisCoordinator
from src.utils.benchmark_utils import calculate_metrics, per_smell_breakdown
from src.evaluation import EvaluationFramework
from src.error_analysis import ErrorAnalyzer
```

---

## 📝 Notes

1. **Phase Dependencies**: Scripts are ordered by execution phase
   - Phase 1-2: Infrastructure setup (DB, models)
   - Phase 3: Data prep → Experiments
   - Phase 4: Analysis → Paper writing

2. **Data Flow**: 
   - Raw data → `data/` scripts → ChromaDB + JSON
   - JSON → `experiments/` scripts → Predictions
   - Predictions → `analysis/` scripts → Tables + Figures

3. **Configuration**:
   - Experiment config: `exp/{baseline,rag_experiments}/config.json`
   - Ablation config: `exp/ablation_studies/config.json`
   - System config: `config.py`

4. **Reproducibility**:
   - All scripts use fixed seed (seed=42 in configs)
   - Results saved with timestamps
   - Detailed logs in `results/logs/`

---

**Last Updated**: March 11, 2026
**Maintained by**: Code Smell Detection Project Team
