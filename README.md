# LLM-Based Code Smell Detection System
## Empirical Evaluation of Multi-Agent LLM Code Analysis

**Status:** Core System Implemented ✅ — Evaluation Phase In Progress 🔬  
**Project Start:** February 9, 2026  
**Submission Deadline:** May 26, 2026  
**Current Branch:** `Feat/orm`

---

## 📋 Project Overview

An empirical research project evaluating Large Language Models (LLMs) for detecting and explaining code smells. The system uses a multi-agent architecture with a LangGraph state machine, RAG-augmented analysis, and a SQLAlchemy ORM backend for experiment tracking.

- **Core Engine:** LangGraph state machine + LangChain ReAct agents
- **LLM Runtime:** Local Ollama (privacy-preserving, no API costs)
- **RAG Pipeline:** ChromaDB vector store + sentence-transformers embeddings
- **Database:** SQLite + SQLAlchemy 2.0 ORM (9 tables, Alembic migrations)
- **Code Analysis:** tree-sitter parsing, radon/lizard metrics
- **Optional API:** FastAPI (deployment) / Streamlit (UI)
- **Focus:** Privacy-preserving, reproducible, cost-effective code analysis

---

## 🎯 Research Goals

### Primary Research Questions

1. **RQ1:** How accurately can local LLMs detect code smells vs. traditional tools?
2. **RQ2:** Does RAG improve LLM detection accuracy?
3. **RQ3:** Which smell types are LLMs most/least effective at detecting?
4. **RQ4:** How do LLM explanations compare to traditional tool outputs?

### Expected Contributions

- **Empirical:** First rigorous evaluation of local LLMs on MaRV dataset
- **Tool:** Open-source, privacy-preserving code review system
- **Dataset:** Enhanced with LLM predictions and explanations
- **Practical:** Reproducible alternative to commercial APIs

---

## 📁 Project Structure

```
code-smell/
├── src/                               # Core source code (implemented ✅)
│   ├── workflow_graph.py              # LangGraph state machine (main entry)
│   ├── analysis_coordinator.py        # Multi-agent coordination + chunking
│   ├── code_analysis_workflow.py      # Workflow execution + file I/O
│   ├── code_smell_detector.py         # LangChain ReAct agent (detector)
│   ├── rag_pipeline.py                # RAG pipeline orchestration
│   ├── rag_retriever.py               # RAG retriever agent
│   ├── embedding_service.py           # sentence-transformers embeddings
│   ├── vector_store.py                # ChromaDB vector store interface
│   ├── code_parser.py                 # Language detection, AST, metrics
│   ├── code_chunker.py                # Code chunking strategies
│   ├── llm_client.py                  # Ollama LLM client wrapper
│   ├── prompt_templates.py            # LLM prompt templates
│   ├── response_parser.py             # LLM response parser
│   ├── quality_validator.py           # Finding validation + confidence scoring
│   ├── database_manager.py            # SQLAlchemy ORM (9 tables)
│   ├── logger.py                      # Structured JSON logging
│   └── common.py                      # Shared utilities, dataclasses
│
├── alembic/                           # Database migrations
│   └── versions/001_initial_schema.py # Initial ORM schema
│
├── data/                              # Datasets (populate before running)
│   ├── datasets/
│   │   ├── marv/                      # MaRV dataset (download required)
│   │   ├── qualitas_corpus/           # Qualitas corpus (optional)
│   │   └── smelly_code/               # Additional smelly code samples
│   └── ground_truth/                  # Ground truth labels
│
├── results/                           # Experiment outputs
│   ├── exports/                       # CSV/JSON exports (exp_001 complete)
│   ├── predictions/                   # Predictions by strategy
│   │   ├── baseline/
│   │   ├── llm_rag/
│   │   └── llm_vanilla/
│   ├── logs/                          # Run logs (JSON)
│   ├── confusion_matrices/
│   ├── figures/
│   ├── metrics/
│   └── tables/
│
├── exp/                               # Experiment configurations
│   ├── baseline/
│   ├── rag_experiments/
│   └── ablation_studies/
│
├── cache/                             # Embedding cache (auto-populated)
│   └── embeddings/
│
├── chromadb_store/                    # ChromaDB vector data (auto-populated)
│
├── docs/                              # Documentation
│   ├── architecture/                  # System, LLM, Backend, Database docs
│   ├── database/                      # ORM migration guide + cheat sheets
│   ├── planning/                      # WBS, project plan
│   ├── research/                      # Research proposal, literature review
│   ├── deployment/                    # Docker deployment guide
│   └── design/                        # UI/UX specifications
│
├── scripts/                           # Utility scripts
│   └── validate_config.py
│
├── visualization/                     # Visualization utilities
├── paper/                             # Paper drafts
│
├── alembic.ini                        # Alembic configuration
├── config.py                          # Central system configuration
├── docker-compose.yml                 # Docker orchestration
├── Dockerfile                         # Container configuration
├── Makefile                           # Build automation
└── requirements.txt                   # Python dependencies
```

---

##  Quick Start

### Prerequisites
- Python 3.14+
- [Ollama](https://ollama.ai/) installed and running locally
- 16 GB RAM recommended (8 GB minimum for 8B models)

### Setup

```bash
# Clone repository
git clone https://github.com/bibekgupta3333/code-smell.git
cd code-smell

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Pull default LLM model (llama3:8b)
ollama pull llama3:8b

# Initialize the database (creates SQLite + runs Alembic migrations)
alembic upgrade head
```

### Run Analysis

```python
from src.workflow.code_analysis_workflow import CodeAnalysisWorkflow

workflow = CodeAnalysisWorkflow()
results = workflow.analyze_file("path/to/your/code.py")
print(results)
```

### Docker (Optional)

```bash
docker compose up -d
```

---

## 🏗️ Tech Stack

### Core (Implemented)
| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.14+ | Programming language |
| Ollama | 0.6.1 | Local LLM runtime (llama3:8b default) |
| LangChain | 1.2.10 | LLM orchestration |
| LangGraph | 1.0.10 | Workflow state machine |
| ChromaDB | 1.0.0 | Vector store for RAG |
| sentence-transformers | 5.2.3 | Code embeddings |
| SQLAlchemy | 2.0.48 | ORM (9 tables) |
| Alembic | 1.18.4 | Database migrations |
| SQLite | — | Experiment tracking database |
| tree-sitter | 0.25.2 | AST-based code parsing |
| radon / lizard | 6.0.1 / 1.21.2 | Cyclomatic complexity, LOC metrics |
| Pydantic | 2.12.5 | Data validation |
| structlog | 25.5.0 | Structured JSON logging |

### Optional (Deployment / UI)
| Technology | Version | Purpose |
|-----------|---------|---------|
| FastAPI | 0.115.9 | REST API (when needed) |
| Streamlit | 1.54.0 | Web UI |
| Flask | 3.1.3 | Lightweight API alternative |
| Docker | — | Containerization |

### LLM Models (via Ollama)
| Model | Parameters | Use Case |
|-------|-----------|----------|
| llama3:8b | 8B | Default — fast, efficient on M4 Pro |
| llama3:13b | 13B | Higher accuracy (slower) |
| codellama:7b | 7B | Code-specialized tasks |
| mistral:7b | 7B | Fast inference alternative |

---

## 📊 Current Status (March 8, 2026)

### ✅ Implemented & Merged to `main`

**Multi-Agent System (`Feat/multi-agent` → merged)**
- LangGraph state machine workflow (`src/workflow_graph.py`)
- LangChain ReAct detector agent (`src/code_smell_detector.py`)
- Multi-agent coordination with chunking (`src/analysis_coordinator.py`)
- RAG pipeline with ChromaDB (`src/rag_pipeline.py`, `src/rag_retriever.py`)
- Code parsing: language detection, AST, metrics (`src/code_parser.py`)
- Quality validation + confidence scoring (`src/quality_validator.py`)
- Structured JSON logging (`src/logger.py`)

**RAG Pipeline (`Feat/rag-llm-client` → merged)**
- Ollama LLM client wrapper (`src/llm_client.py`)
- Embedding service with caching (`src/embedding_service.py`)
- ChromaDB vector store interface (`src/vector_store.py`)
- Response parser for LLM output (`src/response_parser.py`)
- Prompt templates for code smell detection (`src/prompt_templates.py`)

**ORM & Database (`Feat/orm` — current branch)**
- SQLAlchemy 2.0 ORM with 9 tables (`src/database_manager.py`)
- Alembic migrations (`alembic/versions/001_initial_schema.py`)
- Experiment tracking: agents, requests, responses, findings, ground truth
- All 261 type errors resolved ✅
- ORM documentation with cheat sheets (`docs/database/ORM_MIGRATION_GUIDE.md`)

### 🔬 In Progress
- [ ] Download and index MaRV dataset (`data/datasets/marv/`)
- [ ] Run baseline experiments (no RAG)
- [ ] Run RAG-augmented experiments
- [ ] Evaluate detection accuracy vs. SonarQube/PMD
- [ ] Ablation studies

### 📋 Pending
- [ ] FastAPI deployment layer
- [ ] Streamlit evaluation dashboard
- [ ] Paper writing

---

## 🗄️ Database Schema (ORM)

9 tables tracking the full experiment lifecycle:

| Table | Purpose |
|-------|---------|
| `agents` | Registered LLM agents (role, model) |
| `agent_requests` | LLM prompt requests per agent |
| `agent_responses` | LLM responses with token counts |
| `agent_actions` | Tool invocations by agents |
| `processes` | File analysis processes |
| `experiments` | Experiment runs with configuration |
| `analysis_runs` | Per-snippet analysis results |
| `code_smell_findings` | Detected smells with confidence + severity |
| `ground_truth` | Known labels for evaluation |

See [ORM Migration Guide](docs/database/ORM_MIGRATION_GUIDE.md) for schema details, cheat sheets, and transaction patterns.

---

## 📚 Documentation

### Research
- [Research Proposal](docs/research/RESEARCH_PROPOSAL.md) — Problem statement, RQs, methodology
- [Similar Papers & Gaps](docs/research/SIMILAR_PAPERS_AND_GAPS.md) — Literature review

### Architecture
- [System Architecture](docs/architecture/SYSTEM_ARCHITECTURE.md) — High-level design
- [LLM Architecture](docs/architecture/LLM_ARCHITECTURE.md) — RAG, prompts, LangGraph
- [Backend Architecture](docs/architecture/BACKEND_ARCHITECTURE.md) — API, services
- [Database Architecture](docs/architecture/DATABASE_ARCHITECTURE.md) — Data storage

### Baseline & Benchmarking
- [Baseline Tools Guide](docs/baseline/BASELINE_GUIDE.md) — Installation, running, tools reference, troubleshooting

### Development
- [ORM Migration Guide](docs/database/ORM_MIGRATION_GUIDE.md) — SQLAlchemy patterns, cheat sheets, transactions
- [Deployment Guide](docs/deployment/DEPLOYMENT_GUIDE.md) — Docker setup
- [WBS](docs/planning/WBS.md) — Project timeline and task tracking

### Design
- [Figma Design Prompt](docs/design/FIGMA_DESIGN_PROMPT.md) — UI/UX specifications

---

## 🔬 Dataset

**MaRV - Manually Validated Refactoring Dataset** (primary)
- **Source:** https://github.com/HRI-EU/SmellyCodeDataset
- **Language:** Java
- **Content:** Manually validated code smell examples
- **Smell Types:** Long Method, Large Class, Feature Envy, Data Clumps, etc.
- **Use:** Ground truth for empirical evaluation

**Additional Datasets**
- `data/datasets/qualitas_corpus/` — Large-scale Java software corpus
- `data/datasets/smelly_code/` — Additional smelly code samples

---

## 📈 Success Metrics

| Metric | Target |
|--------|--------|
| F1-Score (overall) | ≥ 75% |
| RAG vs. Vanilla improvement | +10–15% |
| Precision per smell type | tracked |
| Recall per smell type | tracked |
| Response time | < 30s per snippet |
| Code coverage | ≥ 80% |

---

## 🛠️ Development Workflow

### Git Branches
| Branch | Purpose |
|--------|---------|
| `main` | Stable, merged features |
| `Feat/orm` | ORM + DB tracking (current) |
| `Feat/multi-agent` | Multi-agent system (merged) |
| `Feat/rag-llm-client` | RAG + LLM client (merged) |

### Conventional Commits
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code restructuring
- `test:` Tests
- `chore:` Maintenance

### Code Style
- Python 3.14+ with full type hints
- `black` for formatting, `isort` for imports
- Lazy logging: `logger.info("msg: %s", var)` — not f-strings
- Specific exceptions: `SQLAlchemyError`, `ValueError` — not bare `Exception`
- All public APIs documented

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Validate configuration
python scripts/validate_config.py
```

**Target Coverage:** 80%+

---

## 🚧 Known Limitations

- Java-only dataset (MaRV); Python support for local testing only
- Single-file analysis (no cross-file smell detection yet)
- Local LLM accuracy below commercial APIs for nuanced smells
- SQLite not suitable for large-scale parallel experiments (use Postgres for scale)

---

## 🤝 Contributing

### Pull Request Process
1. Branch from `main`: `git checkout -b feat/your-feature`
2. Make changes with type hints + tests
3. Run `black . && isort . && pytest`
4. Submit PR with description
5. Merge after review approval

---

## 📮 Contact

**Repository:** https://github.com/bibekgupta3333/code-smell  
**Project:** Code Smell Detection with LLMs  
**Branch:** `Feat/orm`

---

## 🙏 Acknowledgments

- **MaRV Dataset:** HRI-EU for the manually validated dataset
- **Ollama:** Local LLM deployment runtime
- **LangChain / LangGraph:** Workflow orchestration framework
- **ChromaDB:** Vector store for RAG
- **SQLAlchemy:** Python ORM framework

---

## 📅 Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Planning & Documentation | ✅ Complete |
| Phase 2 | Core System Implementation | ✅ Complete |
| Phase 3 | ORM & Experiment Tracking | 🔄 In Progress |
| Phase 4 | Dataset Indexing & Experiments | ⏳ Pending |
| Phase 5 | Evaluation & Analysis | ⏳ Pending |
| Phase 6 | Paper Writing & Finalization | ⏳ Pending |

---

## 🔗 Important Links

- **MaRV Dataset:** https://github.com/HRI-EU/SmellyCodeDataset
- **Ollama:** https://ollama.ai/
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **SQLAlchemy Docs:** https://docs.sqlalchemy.org/
- **ChromaDB Docs:** https://docs.trychroma.com/

---

**Last Updated:** March 8, 2026  
**Version:** 2.0  
**Status:** Core System Implemented ✅ | Evaluation Phase Starting 🔬
