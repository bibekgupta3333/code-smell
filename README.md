# LLM-Based Code Review System
## Empirical Evaluation of Code Smell Detection

**Status:** Planning & Documentation Phase Complete ✅  
**Project Start:** February 9, 2026  
**Submission Deadline:** May 26, 2026  
**Research Proposal Due:** February 12, 2026

---

## 📋 Project Overview

An empirical research project evaluating Large Language Models (LLMs) for detecting and explaining code smells using the MaRV (Manually Validated Refactoring) dataset. The system combines:

- **Backend:** FastAPI + RAG + Local Ollama + LangGraph
- **Frontend:** Streamlit web interface
- **Vector Store:** ChromaDB (local)
- **Deployment:** Docker containers
- **Focus:** Privacy-preserving, cost-effective code analysis

---

## 🎯 Research Goals

### Primary Research Questions

1. **RQ1:** How accurately can local LLMs detect code smells vs. traditional tools?
2. **RQ2:** Does RAG improve LLM detection accuracy?
3. **RQ3:** Which smell types are LLMs most/least effective at detecting?
4. **RQ4:** How do LLM explanations compare to traditional tool outputs?

### Expected Contributions

- **Empirical:** First rigorous evaluation of local LLMs on MaRV dataset
- **Tool:** Open-source code review system
- **Dataset:** Enhanced with LLM predictions and explanations
- **Practical:** Privacy-preserving alternative to commercial APIs

---

## 📁 Project Structure (Monorepo)

```
code-smell/
├── docs/                          # Documentation
│   ├── architecture/              # System, LLM, Backend, Database architecture
│   ├── planning/                  # WBS, project plan
│   ├── research/                  # Research proposal, papers, gaps
│   ├── deployment/                # Deployment guide
│   └── design/                    # UI/UX design specs
│
├── backend/                       # FastAPI backend
│   ├── apps/                      # API routes, services
│   │   ├── api/                   # FastAPI routes
│   │   └── service/               # Business logic
│   ├── core/                      # Configuration, settings
│   ├── services/                  # LLM, RAG, vector store integrations
│   ├── models/                    # Request/Response models
│   ├── utils/                     # Utilities
│   ├── repositories/              # Data access layer
│   ├── workflows/                 # LangGraph workflows
│   └── tests/                     # Unit & integration tests
│
├── frontend/                      # Streamlit frontend
│   ├── pages/                     # Multi-page app
│   ├── components/                # Reusable UI components
│   └── utils/                     # Frontend utilities
│
├── docker/                        # Docker configurations
│   ├── backend/
│   ├── frontend/
│   └── nginx/
│
├── storage/                       # Data storage (gitignored)
│   ├── chromadb/                  # Vector store data
│   ├── uploads/                   # Temporary uploads
│   ├── results/                   # Analysis results
│   └── datasets/                  # MaRV dataset
│
├── scripts/                       # Utility scripts
│   ├── init_database.py           # Initialize vector store
│   ├── index_marv.py              # Index MaRV dataset
│   └── backup.py                  # Backup data
│
├── docker-compose.yml             # Docker orchestration
├── Makefile                       # Build automation
├── .env.example                   # Environment template
├── .editorconfig                  # Editor configuration
├── .cursorrules                   # Cursor AI rules
├── .prettierrc                    # Code formatting
└── README.md                      # This file
```

---

## 📚 Documentation Completed

### ✅ Research Documentation
- [x] [Research Proposal](docs/research/RESEARCH_PROPOSAL.md) - Complete proposal answering all milestone requirements
- [x] [Similar Papers & Gaps](docs/research/SIMILAR_PAPERS_AND_GAPS.md) - Literature review and gap analysis

### ✅ Planning Documentation
- [x] [Work Breakdown Structure (WBS)](docs/planning/WBS.md) - Complete project plan with status tracking
- [x] Project ideas and requirements captured

### ✅ Architecture Documentation
- [x] [System Architecture](docs/architecture/SYSTEM_ARCHITECTURE.md) - High-level system design
- [x] [LLM Architecture](docs/architecture/LLM_ARCHITECTURE.md) - RAG pipeline, prompt engineering, LangGraph workflows
- [x] [Backend Architecture](docs/architecture/BACKEND_ARCHITECTURE.md) - FastAPI service layer, API design
- [x] [Database Architecture](docs/architecture/DATABASE_ARCHITECTURE.md) - Vector store, cache, file storage

### ✅ Deployment Documentation
- [x] [Deployment Guide](docs/deployment/DEPLOYMENT_GUIDE.md) - Step-by-step Docker deployment

### ✅ Design Documentation
- [x] [Figma Design Prompt](docs/design/FIGMA_DESIGN_PROMPT.md) - Complete UI/UX specifications

### ✅ Development Standards
- [x] `.editorconfig` - Editor consistency
- [x] `.prettierrc` - Code formatting
- [x] `.cursorrules` - Cursor AI coding standards

---

## 🚀 Quick Start (Coming Soon)

### Prerequisites
- Docker 24.0+
- Docker Compose 2.24+
- 16 GB RAM minimum
- 50 GB free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/code-review-llm.git
cd code-review-llm

# Copy environment configuration
cp .env.example .env

# Start services
docker compose up -d

# Pull LLM models
docker exec -it codereview-ollama ollama pull llama3:8b

# Initialize database
docker exec -it codereview-backend python scripts/init_database.py

# Access application
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000/api/docs
```

---

## 🏗️ Tech Stack

### Backend
| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Programming language |
| FastAPI | 0.109+ | Web framework |
| Ollama | Latest | Local LLM runtime |
| LangGraph | 0.0.20+ | Workflow orchestration |
| sentence-transformers | 2.3+ | Embeddings |
| ChromaDB | 0.4.22+ | Vector store |
| Redis | 7.2+ | Caching (optional) |

### Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| Streamlit | 1.30+ | Web UI framework |
| Plotly | 5.18+ | Visualizations |
| Pandas | 2.1+ | Data manipulation |

### LLM Models
| Model | Parameters | Use Case |
|-------|-----------|----------|
| Llama 3 | 8B/13B | Primary analysis |
| CodeLlama | 7B/13B | Code-specialized tasks |
| Mistral | 7B | Fast inference |

### DevOps
| Technology | Purpose |
|-----------|---------|
| Docker | Containerization |
| Docker Compose | Multi-container orchestration |
| GitHub Actions | CI/CD (future) |

---

## 📊 Current Status (Week 1)

### Completed Tasks (14/130) - 10.8%

**Phase 1: Planning & Setup** - 61.5% Complete
- ✅ Research proposal document
- ✅ Work breakdown structure
- ✅ System architecture design
- ✅ LLM architecture design
- ✅ Backend architecture design
- ✅ Database architecture design
- ✅ Editor configuration (.editorconfig)
- ✅ Cursor rules (.cursorrules)
- ✅ Similar papers research
- ✅ Research gap analysis
- ✅ Deployment guide
- ✅ Figma design prompt

### Next Steps (Week 2-3)
- [ ] Monorepo structure setup
- [ ] Docker configuration implementation
- [ ] FastAPI project initialization
- [ ] Ollama local setup
- [ ] ChromaDB configuration
- [ ] Download MaRV dataset

See [WBS](docs/planning/WBS.md) for detailed timeline and tasks.

---

## 🎓 Research Milestone 1 Deliverables

**Due:** February 12, 2026

### ✅ Completed
1. **Team Information** - In research proposal
2. **Problem Statement** - Concrete gap identified  
3. **Research Questions** - 4 primary RQs defined
4. **Scope & Assumptions** - Clear boundaries set
5. **Initial Methodology** - High-level approach documented
6. **Expected Outcomes** - Tool + empirical findings + dataset

**Submission Ready:** All requirements met in [Research Proposal](docs/research/RESEARCH_PROPOSAL.md)

---

## 🔬 Dataset

**MaRV - Manually Validated Refactoring Dataset**
- **Source:** https://github.com/HRI-EU/SmellyCodeDataset
- **Language:** Java
- **Content:** Manually validated code smell examples
- **Smell Types:** Long Method, Large Class, Feature Envy, Data Clumps, etc.
- **Use:** Ground truth for empirical evaluation

---

## 👥 Team Roles (To Be Assigned)

- **Project Lead:** Overall coordination, backend development
- **ML/LLM Engineer:** RAG implementation, LLM integration
- **Frontend Developer:** Streamlit UI, UX
- **Research Lead:** Evaluation, documentation, analysis

---

## 📖 Key Documents

### Must Read (Getting Started)
1. [Research Proposal](docs/research/RESEARCH_PROPOSAL.md) - Understand project goals
2. [WBS](docs/planning/WBS.md) - See timeline and tasks
3. [System Architecture](docs/architecture/SYSTEM_ARCHITECTURE.md) - Understand high-level design
4. [Deployment Guide](docs/deployment/DEPLOYMENT_GUIDE.md) - Setup instructions

### Reference (During Development)
- [LLM Architecture](docs/architecture/LLM_ARCHITECTURE.md) - RAG, prompts, workflows
- [Backend Architecture](docs/architecture/BACKEND_ARCHITECTURE.md) - API, services
- [Database Architecture](docs/architecture/DATABASE_ARCHITECTURE.md) - Data storage
- [Cursor Rules](.cursorrules) - Coding standards

### Design
- [Figma Design Prompt](docs/design/FIGMA_DESIGN_PROMPT.md) - UI/UX specifications

---

## 🛠️ Development Workflow

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/smell-detection-service

# Make changes, commit
git add .
git commit -m "feat: implement smell detection service"

# Push and create PR
git push origin feature/smell-detection-service
```

### Conventional Commits
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Formatting
- `refactor:` Code restructuring
- `test:` Tests
- `chore:` Maintenance

### Code Review
- All changes via pull requests
- Minimum 1 reviewer approval
- Pass all tests
- Follow `.cursorrules` standards

---

## 🧪 Testing Strategy

### Unit Tests
- Backend services
- Utility functions
- Data models

###  Integration Tests
- API endpoints
- LLM integration
- Vector store operations

### End-to-End Tests
- Full analysis workflow
- Frontend-backend integration

**Target Coverage:** 80%+

---

## 📈 Success Metrics

### Technical Metrics
- **Detection Accuracy:** F1-Score ≥ 75%
- **RAG Improvement:** +10-15% over vanilla LLM
- **Response Time:** < 30s for typical analysis
- **System Uptime:** ≥ 99%

### Research Metrics
- **Precision:** per smell type
- **Recall:** per smell type
- **F1-Score:** overall and per smell
- **Comparison:** LLM vs. SonarQube vs. PMD

### Project Metrics
- **Code Coverage:** ≥ 80%
- **Documentation:** 100% of public APIs
- **Timeline Adherence:** ≥ 90% tasks on time

---

## 🚧 Known Limitations

### Scope Limitations
- Java-only (primary focus)
- MaRV dataset size constraints
- No real-time IDE integration (standalone web app)
- No automatic refactoring (detection only)

### Technical Constraints
- Local LLM performance vs. commercial APIs
- Vector store scalability (small to medium datasets)
- Single-file analysis (no cross-file smell detection)

### Resource Constraints
- Limited compute for large models
- No budget for commercial APIs
- 12-week project timeline

---

## 🤝 Contributing

(To be populated when development starts)

### Code Style
- Follow `.cursorrules`
- Use `.editorconfig` settings
- Run `black` for Python formatting
- Type hints required

### Pull Request Process
1. Create feature branch
2. Make changes
3. Write/update tests
4. Update documentation
5. Submit PR with description
6. Address review comments
7. Merge after approval

---

## 📝 License

(To be determined - likely MIT or Apache 2.0 for open-source contribution)

---

## 📮 Contact

**Project:** Code Smell Detection with LLMs  
**Institution:** [Your University]  
**Course:** [Course Code & Name]  
**Instructor:** [Instructor Name]

---

## 🙏 Acknowledgments

- **MaRV Dataset:** HRI-EU for the manually validated dataset
- **Ollama:** For local LLM deployment
- **FastAPI:** Modern Python web framework
- **Streamlit:** Rapid UI development
- **ChromaDB:** Vector store solution

---

## 📅 Timeline Summary

| Phase | Weeks | Completion |
|-------|-------|------------|
| **Phase 1:** Planning & Setup | 1-2 | 61.5% ✅ |
| **Phase 2:** Backend Development | 3-6 | 0% ⏳ |
| **Phase 3:** Frontend Development | 7-8 | 4.8% ⏳ |
| **Phase 4:** Evaluation & Research | 9-10 | 0% ⏳ |
| **Phase 5:** Testing & Integration | 11 | 0% ⏳ |
| **Phase 6:** Documentation & Finalization | 12 | 8.3% ⏳ |

**Overall Progress:** 10.8% (14/130 tasks)

---

## 🔗 Important Links

- **MaRV Dataset:** https://github.com/HRI-EU/SmellyCodeDataset
- **Ollama:** https://ollama.ai/
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Streamlit Docs:** https://docs.streamlit.io/
- **ChromaDB Docs:** https://docs.trychroma.com/
- **LangGraph:** https://langchain-ai.github.io/langgraph/

---

**Last Updated:** February 9, 2026  
**Version:** 1.0  
**Status:** Documentation Complete, Ready for Development ✅
