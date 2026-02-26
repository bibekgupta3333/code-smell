# LLM-Based Code Smell Detection - Project Structure

## Directory Organization

```
code-smell/
├── config.py              # Main configuration file (Ollama, ChromaDB, paths)
├── requirements.txt       # Python dependencies (pinned versions)
├── venv/                  # Python virtual environment (git-ignored)
│
├── src/                   # Core Python modules
│   ├── llm_client.py
│   ├── rag_pipeline.py
│   ├── analysis_coordinator.py
│   └── ...
│
├── data/                  # Datasets and ground truth
│   ├── datasets/
│   │   ├── marv/         # MaRV dataset (expert-validated)
│   │   ├── qualitas_corpus/
│   │   └── smelly_code/
│   └── ground_truth/      # Manually verified examples
│
├── exp/                   # Experiment results
│   ├── baseline/          # Vanilla LLM experiments
│   ├── rag_experiments/   # RAG-enhanced experiments
│   └── ablation_studies/  # Hyperparameter tuning
│
├── results/               # Benchmark results for paper
│   ├── predictions/       # Tool predictions
│   │   ├── baseline/      # SonarQube, PMD, etc.
│   │   ├── llm_vanilla/   # Vanilla LLM
│   │   └── llm_rag/       # RAG-enhanced LLM
│   ├── confusion_matrices/
│   ├── performance/       # Latency, throughput logs
│   ├── resources/         # CPU, memory profiles
│   ├── figures/           # Plots for paper
│   ├── tables/            # LaTeX/CSV tables
│   ├── metrics/           # LLM-specific metrics
│   │   ├── hallucination_rate.csv
│   │   ├── cache_performance.csv
│   │   └── confidence_scores.csv
│   └── logs/              # Experiment logs
│
├── scripts/               # Utility scripts
│   ├── run_experiment.py
│   ├── run_baseline_tools.py
│   ├── index_datasets.py
│   └── analyze_results.py
│
├── visualization/         # Web-based visualization (optional)
│   └── app.py
│
├── paper/                 # LaTeX source files
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
├── chromadb_store/        # Vector database (git-ignored)
├── cache/                 # LLM response cache (git-ignored)
└── code_smell_experiments.db  # SQLite database
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 2. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Initialize Configuration

```bash
python config.py  # Creates all directories and verifies setup
```

### 4. Verify Installation

```bash
python -c "import ollama, chromadb, langchain; print('✓ All imports successful')"
```

## Configuration

Edit `config.py` to customize:

- **Ollama URL**: Default `http://localhost:11434`
- **Default Model**: `llama3:8b` (optimized for M4 Pro)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **RAG Parameters**: top_k, similarity_threshold, MMR lambda
- **Performance Targets**: latency, throughput, memory

## Environment Variables

Create a `.env` file (optional):

```bash
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3:8b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOG_LEVEL=INFO
```

## Hardware Requirements

**Optimized for M4 Pro:**
- RAM: 16GB+ (4-8GB for model, rest for ChromaDB)
- Storage: 2GB+ for models and vector store
- CPU: Apple Silicon (efficient inference)
- GPU: Not required (CPU inference)

## Research Workflow

1. **Setup** (Phase 1): Environment and datasets
2. **Development** (Phase 2): Implement core modules
3. **Experiments** (Phase 3): Run baseline and RAG experiments
4. **Evaluation** (Phase 4): Analyze results, statistical tests
5. **Paper** (Phase 5): Generate figures/tables, write paper
6. **Optional** (Phase 6): Deploy demo system

## Key Files

- `config.py`: All configuration in one place
- `requirements.txt`: Pinned dependencies for reproducibility
- `README.md`: Project documentation
- `.cursorrules`: Cursor AI coding standards
- `.editorconfig`: Editor configuration

## Git Ignored

- `venv/`: Virtual environment
- `chromadb_store/`: Vector database
- `cache/`: LLM response cache
- `*.db`: SQLite databases
- `__pycache__/`, `*.pyc`

## Next Steps

See [docs/planning/WBS.md](docs/planning/WBS.md) for detailed work breakdown structure.
