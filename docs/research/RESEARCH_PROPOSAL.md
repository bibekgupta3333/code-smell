# Research Proposal: LLM-Based Code Review System
## Empirical Evaluation of LLMs for Code Smell Detection

**Project Type:** Selected Project from Course List  
**Submitted:** February 9, 2026  
**Due Date:** February 12, 2026

---

## 1. Team Information

**Project Title:** Empirical Evaluation of LLM-Based Code Analysis Using Smelly Code Dataset with RAG Enhancement

**Selected Project:** Project 1 - Empirical Evaluation of LLM-Based Code Analysis Using Smelly Code Dataset

**Team Members:**
- [Team Member 1] - Project Lead, Backend Development
- [Team Member 2] - ML/LLM Integration, RAG Implementation
- [Team Member 3] - Frontend Development, UI/UX
- [Team Member 4] - Research, Evaluation, Documentation

---

## 2. Problem Statement

### What Problem Are We Addressing?

Software quality degradation due to code smells represents a persistent challenge in software engineering. Code smells—indicators of poor design or implementation choices—can lead to increased technical debt, reduced maintainability, and higher defect rates. While traditional static analysis tools exist, they suffer from several limitations:

1. **High False Positive Rates:** Existing tools generate numerous false alarms, leading to developer fatigue
2. **Limited Contextual Understanding:** Traditional tools lack semantic understanding of code intent and context
3. **Rigid Rule-Based Detection:** Cannot adapt to evolving coding patterns and emerging smells
4. **Poor Explainability:** Provide limited reasoning about why something is flagged as a smell
5. **Limited Learning Capability:** Cannot leverage historical code reviews and developer feedback

Large Language Models (LLMs) have shown promise in code understanding tasks, but their effectiveness in detecting and explaining code smells remains under-explored, particularly when:
- Using locally-deployable open-source models (privacy and cost considerations)
- Enhanced with Retrieval-Augmented Generation (RAG) for improved accuracy
- Evaluated against manually validated ground truth datasets
- Compared systematically with traditional static analysis tools

### Why It Matters

**Technical Impact:**
- Advances understanding of LLM capabilities in software quality assurance
- Provides empirical evidence for LLM effectiveness vs. traditional approaches
- Identifies specific code smell types where LLMs excel or struggle
- Demonstrates viability of local LLM deployment for code analysis

**Educational Impact:**
- Creates learning resource for software engineering students
- Demonstrates practical application of RAG and LLMs in SE domain
- Provides hands-on experience with modern AI tools in code analysis

**Practical Impact:**
- Enables cost-effective, privacy-preserving code review automation
- Reduces developer time spent on manual code reviews
- Improves code quality through more accurate smell detection
- Provides actionable, explainable recommendations for developers

### Who Benefits

1. **Software Developers:** Get better code review feedback with explanations
2. **Development Teams:** Reduce technical debt through improved smell detection
3. **Researchers:** Gain empirical data on LLM effectiveness in code analysis
4. **Students:** Learn about code quality and modern AI applications in SE
5. **Organizations:** Access privacy-preserving, cost-effective code analysis tools

### Concrete Gap or Limitation

**Existing Research Gaps:**

1. **Lack of Local LLM Evaluation:** Most studies use commercial APIs (GPT-4, Claude), limited work on locally-deployable open-source models (Llama, CodeLlama, Mistral)

2. **Missing RAG Integration:** Current approaches don't leverage RAG to enhance LLM performance with code smell knowledge bases and historical examples

3. **Insufficient Empirical Validation:** Limited studies using manually validated ground truth datasets (like MaRV dataset) for rigorous evaluation

4. **Unclear Cost-Accuracy Tradeoffs:** No clear guidance on when local LLMs + RAG match or exceed commercial LLM performance

5. **Limited Comparative Analysis:** Few studies systematically compare LLMs against established static analysis tools on same dataset

6. **Explainability Gap:** Limited exploration of LLM's reasoning quality and explanation usefulness

**Our Project Addresses These Gaps By:**
- Using local Ollama models for privacy-preserving analysis
- Implementing RAG with local vector stores for knowledge enhancement
- Rigorous evaluation against MaRV's manually validated dataset
- Systematic comparison with baseline static analysis tools
- Analysis of LLM explanations and reasoning quality
- Open-source implementation for reproducibility

---

## 3. Research Questions and Goals

### Primary Research Questions

**RQ1:** How accurately can local LLMs detect code smells compared to traditional static analysis tools when evaluated on manually validated ground truth data?

**Metrics:** Precision, Recall, F1-Score per smell type

**RQ2:** Does Retrieval-Augmented Generation (RAG) with code smell examples and patterns improve LLM detection accuracy compared to vanilla LLM prompting?

**Metrics:** Accuracy improvement (%), False positive reduction rate

**RQ3:** Which code smell types are local LLMs most/least effective at detecting, and what factors influence detection performance?

**Metrics:** Per-smell performance, Confusion matrix, Error analysis

**RQ4:** How does the quality of LLM-generated explanations compare to traditional tool outputs in terms of actionability and developer understanding?

**Metrics:** Explanation quality scores, Developer comprehension survey

### Secondary Research Questions

**RQ5:** What are the computational costs and latency characteristics of local LLM-based code smell detection compared to traditional tools?

**Metrics:** Inference time, Resource usage, Throughput

**RQ6:** Can LangGraph-based orchestration improve consistency and reliability of LLM-based code analysis?

**Metrics:** Consistency across runs, Error rates

### Goals

1. **Empirical Contribution:** Provide rigorous empirical evidence of local LLM effectiveness in code smell detection

2. **Tool Contribution:** Develop open-source, deployable code review system using FastAPI, Ollama, and RAG

3. **Dataset Contribution:** Create enhanced dataset with LLM predictions and explanations for future research

4. **Practical Contribution:** Demonstrate viable alternative to commercial LLM APIs for code analysis

5. **Educational Contribution:** Create comprehensive system suitable for teaching modern AI in software engineering

---

## 4. Scope & Assumptions

### In Scope

**Technical Scope:**
- Local LLM deployment using Ollama (Llama 3, CodeLlama, Mistral models)
- RAG implementation with local vector store (ChromaDB/FAISS)
- LangGraph orchestration for multi-step analysis
- FastAPI-based backend service
- Streamlit-based web interface
- Docker containerization for deployment
- Free/open-source embedding models (sentence-transformers)

**Research Scope:**
- Code smell detection and classification
- Comparison with baseline static analysis tools (SonarQube, PMD)
- Evaluation on MaRV Smelly Code Dataset
- Precision, Recall, F1-Score metrics
- Per-smell-type analysis
- Explanation quality assessment
- False positive/negative analysis

**Code Smell Types (from MaRV Dataset):**
- Long Method
- Long Parameter List
- Large Class
- Feature Envy
- Data Clumps
- Duplicate Code
- God Class
- Shotgun Surgery
- Divergent Change
- Switch Statements

**Programming Languages:**
- Primary: Java (MaRV dataset focus)
- Secondary: Python (if time permits)

### Out of Scope

**Explicitly Excluded:**
- Automatic code refactoring (only detection and explanation)
- Real-time IDE integration (standalone web app only)
- Multi-repository code analysis
- Distributed deployment across clusters
- Code smell prevention/prediction
- Cross-language smell detection (focus on Java)
- Training custom LLMs (using pre-trained models only)
- Production-grade security features (authentication, authorization)
- Commercial LLM APIs (GPT-4, Claude, etc.)
- Mobile application interface

### Constraints

**Dataset Constraints:**
- Limited to MaRV Smelly Code Dataset (publicly available)
- Ground truth limited to manually validated samples
- Java-centric codebase examples

**Platform Constraints:**
- Development on local machines/university servers
- Docker-based deployment only
- CPU/GPU availability for local LLM inference
- Vector store size limited by available storage

**Model Constraints:**
- Open-source models only (Ollama-compatible)
- Model size limited by available compute resources
- Inference speed constraints for interactive UI

**Time Constraints:**
- 12-week project timeline
- Weekly progress milestones
- Final deliverables by May 26, 2026

**Resource Constraints:**
- No budget for commercial API calls
- Limited to free/open-source tools and libraries
- Team of 4 members with varying expertise

### Assumptions

1. **Model Availability:** Ollama models (Llama 3, CodeLlama) perform adequately for code understanding
2. **Dataset Quality:** MaRV dataset provides sufficient, high-quality ground truth
3. **Compute Resources:** Available compute is sufficient for local LLM inference
4. **RAG Effectiveness:** RAG will improve performance over vanilla prompting
5. **Development Environment:** Team has access to suitable development machines
6. **Tool Availability:** All required open-source tools remain accessible
7. **Scope Feasibility:** Project is achievable within 12-week timeframe

---

## 5. Initial Methodology (High Level)

### Overall Approach

**Type of Study:** Empirical, Quantitative, Comparative

### Phase 1: Setup & Infrastructure (Weeks 1-2)

**Activities:**
- Monorepo project structure setup (backend + frontend)
- Docker environment configuration
- Ollama installation and model selection
- Development environment standardization

**Deliverables:**
- Working development environment
- Project documentation structure
- Initial system design

### Phase 2: Data Preparation (Weeks 2-3)

**Activities:**
- MaRV dataset download and preprocessing
- Code smell annotation extraction
- Data splitting (train/validation/test)
- Vector store indexing preparation

**Dataset:** MaRV - Manually Validated Refactoring Dataset
- Source: https://github.com/HRI-EU/SmellyCodeDataset
- Contains: Java code samples with manually validated code smells
- Annotations: Expert-labeled code smell instances with ground truth

**Deliverables:**
- Preprocessed dataset ready for evaluation
- Data statistics and distribution analysis

### Phase 3: Backend Development (Weeks 3-6)

**Core Components:**

1. **LLM Integration Service:**
   - Ollama client wrapper
   - Model selection logic
   - Prompt engineering for code smell detection
   - Response parsing and structuring

2. **RAG Pipeline:**
   - Embedding model: sentence-transformers (all-MiniLM-L6-v2 or similar)
   - Vector store: ChromaDB or FAISS
   - Code smell pattern indexing
   - Similarity search implementation
   - Context retrieval for LLM prompts

3. **LangGraph Workflow:**
   - Graph-based orchestration of analysis steps
   - Nodes: Code preprocessing → Smell detection → Classification → Reasoning
   - State management for multi-step analysis
   - Error handling and retry logic

4. **FastAPI Backend:**
   - RESTful API endpoints for code submission
   - Code smell detection endpoint
   - Results retrieval endpoint
   - Health check and monitoring

**Architecture Layers:**
```
apps/
  ├── api/       # FastAPI routes & controllers
  ├── service/   # Business logic layer
core/            # Configuration, settings, constants
services/        # External integrations (LLM, embeddings)
utils/           # Helper functions, utilities
```

**Deliverables:**
- Working backend API
- RAG pipeline operational
- LangGraph workflow functional

### Phase 4: Frontend Development (Weeks 7-8)

**Streamlit Application:**

**Pages:**
1. **Code Analysis Dashboard**
   - Code input (paste or file upload)
   - Submit for analysis
   - Real-time results display

2. **Results Visualization**
   - Detected smells list
   - Code highlighting
   - LLM explanations
   - Severity indicators

3. **Comparison View**
   - LLM results vs. baseline tools
   - Side-by-side comparison

4. **Analytics Dashboard**
   - Historical analysis statistics
   - Performance metrics

**Deliverables:**
- Interactive web interface
- Intuitive user experience
- Responsive design

### Phase 5: Evaluation Framework (Weeks 9-10)

**Baseline Tools Setup:**
- SonarQube installation and configuration
- PMD setup for Java analysis
- Automated baseline execution pipeline

**Evaluation Pipeline:**
1. **Automated Testing:**
   - Iterate through MaRV test set
   - Generate predictions for each code sample
   - Store results in structured format

2. **Metrics Calculation:**
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
   - Per-smell-type breakdown

3. **Comparison Analysis:**
   - LLM (vanilla) vs. LLM + RAG vs. Baseline tools
   - Statistical significance testing
   - Confusion matrix generation

4. **Qualitative Analysis:**
   - Explanation quality assessment
   - False positive manual review
   - Error pattern analysis

**Deliverables:**
- Comprehensive evaluation results
- Statistical analysis
- Performance comparison reports

### Phase 6: Integration & Testing (Week 11)

**Activities:**
- End-to-end integration testing
- Docker Compose orchestration
- Performance optimization
- Bug fixes and refinements

**Deliverables:**
- Fully integrated system
- Docker deployment ready
- Test coverage reports

### Phase 7: Documentation & Finalization (Week 12)

**Activities:**
- Final documentation
- User guide creation
- Deployment guide
- Research paper/report writing
- Demo preparation

**Deliverables:**
- Complete documentation
- Final presentation
- Research findings report
- Live demo

### Tools and Technologies Planned

**Backend:**
- Python 3.11+
- FastAPI (web framework)
- Ollama (local LLM runtime)
- LangChain/LangGraph (orchestration)
- sentence-transformers (embeddings)
- ChromaDB or FAISS (vector store)
- Pydantic (validation)

**Frontend:**
- Streamlit (web UI)
- Plotly (visualizations)

**Deployment:**
- Docker & Docker Compose
- GitHub Actions (CI/CD)

**Analysis:**
- SonarQube (baseline)
- PMD (baseline)
- Pandas (data analysis)
- Scikit-learn (metrics)

**Development:**
- Git/GitHub
- VS Code/Cursor
- pytest (testing)

### Type of Analysis

**Quantitative Analysis:**
- Precision, Recall, F1-Score calculation
- Statistical significance testing (t-tests, Mann-Whitney U)
- Performance benchmarking (latency, throughput)
- Resource utilization metrics

**Qualitative Analysis:**
- Explanation quality assessment
- Error pattern categorization
- User experience feedback (if time permits)

**Comparative Analysis:**
- LLM vs. traditional tools head-to-head
- RAG impact quantification
- Per-smell-type effectiveness comparison

---

## 6. Expected Outcomes

### Primary Deliverables

1. **Tool Artifact: LLM-Based Code Review System**
   - Open-source, deployable web application
   - FastAPI backend with RAG integration
   - Streamlit frontend for user interaction
   - Docker-compose deployment configuration
   - Monorepo structure for maintainability
   - Comprehensive documentation

2. **Empirical Findings: Evaluation Report**
   - Quantitative results (precision, recall, F1 per smell)
   - Comparative analysis (LLM vs. baselines)
   - RAG impact assessment
   - Per-smell-type performance breakdown
   - Statistical significance analysis
   - False positive/negative analysis

3. **Dataset Enhancement**
   - MaRV dataset with LLM predictions
   - Explanation corpus for each detection
   - Confidence scores and reasoning chains
   - Available for future research

4. **Research Paper/Technical Report**
   - Problem formulation
   - Methodology description
   - Results and analysis
   - Discussion of findings
   - Limitations and future work
   - Reproducibility guidelines

### Expected Research Outcomes

**Hypothesis:** Local LLMs enhanced with RAG will achieve comparable performance to traditional static analysis tools while providing superior explanations.

**Expected Findings:**

1. **Detection Performance:**
   - LLM + RAG: F1-Score ~75-85% on well-defined smells
   - Vanilla LLM: F1-Score ~60-70%
   - Baseline tools: F1-Score ~70-80%

2. **Smell-Specific Results:**
   - **High LLM Performance:** Long Method, Large Class (structural, measurable)
   - **Medium Performance:** Feature Envy, Data Clumps (contextual reasoning)
   - **Lower Performance:** Shotgun Surgery, Divergent Change (requires codebase-wide context)

3. **RAG Impact:**
   - Expected 10-15% improvement in F1-Score vs. vanilla prompting
   - Reduction in false positives by ~20-30%
   - Improved consistency across similar code patterns

4. **Explanation Quality:**
   - LLM explanations more detailed and actionable than baseline tools
   - Better alignment with developer mental models
   - Clearer reasoning about design principles

### Anticipated Risks and Challenges

**Technical Risks:**

| Risk | Impact | Mitigation Strategy |
|------|--------|-------------------|
| **Ollama model performance insufficient** | High | Test multiple models (Llama 3, CodeLlama, Mistral); implement ensemble if needed |
| **RAG not improving accuracy** | Medium | Experiment with chunking strategies, different embedding models, retrieval parameters |
| **Vector store scalability issues** | Low | Start with small subset, optimize indexing, use efficient similarity search |
| **LLM hallucinations** | High | Implement confidence thresholding, validation logic, RAG grounding |
| **Docker resource constraints** | Medium | Optimize container images, set resource limits, use model quantization |

**Dataset Risks:**

| Risk | Impact | Mitigation Strategy |
|------|--------|-------------------|
| **Insufficient ground truth samples** | Medium | Supplement with additional validated samples if needed |
| **Dataset bias toward certain smells** | Medium | Report findings with dataset distribution; stratified sampling |
| **Java-only limitation** | Low | Accept limitation; note in scope; suggest multi-language as future work |

**Research Risks:**

| Risk | Impact | Mitigation Strategy |
|------|--------|-------------------|
| **Negative results (LLM underperforms)** | Medium | Still valuable finding; focus on error analysis and lessons learned |
| **Baseline tools difficult to configure** | Low | Allocate time for tool setup; use default configurations if needed |
| **Explanation quality hard to measure** | Medium | Define rubric; possibly manual evaluation on subset |

**Project Management Risks:**

| Risk | Impact | Mitigation Strategy |
|------|--------|-------------------|
| **Scope creep** | High | Strict adherence to WBS; regular scope reviews |
| **Integration complexity** | Medium | Incremental integration; early integration testing |
| **Time constraints** | Medium | Prioritize core features; have MVP milestone at week 10 |
| **Team coordination** | Low | Weekly standups; clear role assignment; documentation |

### Success Criteria

**Minimum Viable Outcome (MVP):**
- ✅ Working code review system (deployed via Docker)
- ✅ LLM detection on at least 5 code smell types
- ✅ Evaluation on MaRV dataset subset (≥100 samples)
- ✅ Basic comparison with one baseline tool
- ✅ Documentation and deployment guide

**Target Outcome:**
- ✅ Full system with RAG integration
- ✅ Evaluation on all major smell types in MaRV
- ✅ Comparison with multiple baseline tools
- ✅ Statistical analysis of results
- ✅ Explanation quality assessment
- ✅ Comprehensive documentation
- ✅ Research paper/technical report

**Stretch Goals:**
- Multi-model ensemble approach
- Active learning for continuous improvement
- Developer user study for explanation quality
- Cross-language validation (Python subset)

### Educational Value

**Learning Objectives:**
1. Hands-on experience with modern LLM technologies
2. Understanding of RAG architectures
3. Empirical software engineering research methods
4. Full-stack development (backend + frontend)
5. Docker containerization and deployment
6. Code quality and software maintenance concepts

**Skill Development:**
- FastAPI backend development
- LLM integration and prompt engineering
- Vector database and embeddings
- Streamlit UI development
- Research methodology and evaluation
- Technical writing and presentation

---

## 7. Timeline Summary

| Phase | Duration | Key Milestones |
|-------|----------|----------------|
| Planning & Setup | Weeks 1-2 | Proposal, architecture, environment ready |
| Backend Development | Weeks 3-6 | LLM service, RAG pipeline, API complete |
| Frontend Development | Weeks 7-8 | Streamlit app functional |
| Evaluation | Weeks 9-10 | Empirical results, comparison complete |
| Testing & Integration | Week 11 | System integrated, Docker deployed |
| Documentation & Finalization | Week 12 | All deliverables ready |

**Key Dates:**
- **Feb 12, 2026:** Project proposal due
- **Feb 20, 2026:** Backend infrastructure ready
- **Mar 20, 2026:** LangGraph workflow complete
- **Apr 20, 2026:** Frontend functional
- **May 10, 2026:** Evaluation complete
- **May 26, 2026:** Final project submission

---

## 8. Conclusion

This project addresses a concrete gap in software engineering research by providing rigorous empirical evaluation of local, open-source LLMs for code smell detection, enhanced with RAG techniques. The work combines practical tool development with academic research, delivering both a usable artifact and empirical findings. The project scope is feasible within the 12-week timeframe, with clear deliverables and success criteria. The anticipated outcomes will benefit researchers, developers, and students while contributing to the growing body of knowledge on LLM applications in software engineering.

**Key Contributions:**
1. First comprehensive evaluation of local LLMs + RAG on manually validated code smell dataset
2. Open-source, deployable code review system
3. Empirical evidence for LLM effectiveness vs. traditional tools
4. Enhanced dataset for future research
5. Practical guidance for using local LLMs in code analysis

The project balances technical innovation, research rigor, and practical applicability, making it an ideal candidate for the course project while providing meaningful contributions to the software engineering community.

---

**References:**
- MaRV: A Manually Validated Refactoring Dataset - https://github.com/HRI-EU/SmellyCodeDataset
- Course Project Requirements - Milestone 1 Specification

**Submitted by:** [Team Name]  
**Date:** February 9, 2026
