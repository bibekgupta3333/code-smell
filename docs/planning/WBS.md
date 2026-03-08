# Work Breakdown Structure (WBS)
## LLM-Based Code Review System for Code Smell Detection
## Privacy-Preserving, RAG-Enhanced Local LLM Solution

**Project Duration**: 12 weeks  
**Last Updated**: March 2, 2026  
**Current Status**: 🚀 Phase 2.5 Complete - Database Manager + Agent Tracking Infrastructure  
**Research Status**: 16 papers analyzed, 12 research gaps identified  
**Novelty**: First privacy-preserving, RAG-enhanced local LLM system with Deep Agent orchestration for production code smell detection

**Environment Notes**:
- ✅ Ollama installed locally (no Docker required)
- ✅ Python 3.14 environment ready (upgraded from 3.11)
- ✅ Git repository initialized
- ✅ LangGraph StateGraph workflow implemented
- ✅ LangChain Deep Agent pattern integrated
- ℹ️ **Research Project**: Simple structure for experiments and paper publication
- 🎯 **Dual Track**: Research (paper) + Implementation (working system)

---

## Recent Enhancements (Mar 3, 2026)

**Python 3.14 Compatibility Migration Complete**
- ✅ Updated all dependencies to Python 3.14 compatible versions
- ✅ Fixed dependency conflicts: fastapi (0.135.1 → 0.115.9), pandas (3.0.1 → 2.3.3)
- ✅ Removed incompatible package: tree-sitter-languages (all versions cap at Python <3.13)
- ✅ Updated 45+ package versions for Python 3.14 support
- ✅ Key updates: torch 2.10.0, langchain 1.2.10, pydantic 2.12.5, numpy 2.4.2
- ✅ All packages now support Python 3.14 (verified cp314 wheels)

**Phase 2.5.1 Complete: SQLAlchemy ORM + Alembic Migration Infrastructure**

- ✅ Refactored `src/database_manager.py` from raw SQL → SQLAlchemy ORM (757 lines)
- ✅ Created 9 SQLAlchemy declarative ORM models with proper relationships:
  - Agent, AgentRequest, AgentResponse, AgentAction, Process
  - Experiment, AnalysisRun, CodeSmellFinding, GroundTruth
- ✅ Replaced thread-local SQLite connections with scoped_session pattern
- ✅ Maintained 100% backward compatible API (all method signatures unchanged)
- ✅ Added connection pooling (QueuePool, 5 connections, M4 Pro tuned)
- ✅ Initialized Alembic version control for schema management:
  - `alembic init alembic` - Created migration infrastructure
  - Created `alembic/versions/001_initial_schema.py` - Clean SQLite-compatible migration
  - Successfully applied migration with `alembic upgrade head`
- ✅ All 9 ORM tables created with proper indices and foreign keys
- ✅ Database schema verified (SQLite PRAGMA table_info)
- ✅ Updated in requirements.txt: sqlalchemy==2.0.48, alembic==1.18.4, greenlet==3.3.2 (Python 3.14 compatible)

**Migration Architecture**:
- Uses Alembic for schema versioning and reproducibility
- sqlalchemy.url configured to `sqlite:///./cache/system.db`
- Scoped sessions ensure thread-safe database access
- QueuePool with pool_pre_ping for reliability

**Phase 2.5 Complete: Multi-Agent Database & Experiment Tracking**

- ✅ Created `src/database_manager.py` with SQLite backend
- ✅ Implemented 9 database tables for agent & experiment tracking
- ✅ Full CRUD operations for all agent and analysis data
- ✅ Agent performance statistics and analytics
- ✅ CSV/JSON export for paper results
- ✅ All tests passing, optimized for M4 Pro
- ✅ Integrated into Makefile (`make run-database`)
- ✅ Refactored to SQLAlchemy ORM for better maintainability

---

## Previous Enhancement (Feb 26, 2026)

**Phase 2.4 Enhanced: LangGraph Workflow + LangChain Deep Agents**

- ✅ Created LangGraph StateGraph for workflow orchestration (`src/workflow_graph.py` - 520+ lines)
- ✅ Refactored CodeSmellDetector to LangChain Deep Agent pattern with 4 specialized tools
- ✅ Implemented manual tool orchestration (Structure → Context → Detection → Classification)
- ✅ Full integration with AnalysisCoordinator and existing modules
- ✅ Comprehensive documentation: DEEP_AGENT_REFACTORING.md, LANGRAPH_LANGCHAIN_*.md
- ✅ All tests passing, backward compatible

---

## Agent Architecture

Following a multi-agent pattern similar to recent research frameworks, our system employs specialized agents for different analysis tasks:

### 🎯 Core Agents

**1. Analysis Coordinator (Manager)**
- **Role**: Orchestrates the entire code smell detection workflow
- **Responsibilities**: 
  - Receives code submissions from users/experiments
  - Splits large codebases into analyzable chunks
  - Assigns analysis tasks to detector agents
  - Aggregates results from multiple detectors
  - Manages database tracking of analysis runs
- **Module**: `src/analysis_coordinator.py`
- **System Prompt**: "You are a senior software quality assurance manager specializing in code smell detection and technical debt analysis."

**2. Code Smell Detector (Member) - Deep Agent**
- **Role**: Analyzes code snippets for specific smell types using multi-tool reasoning
- **Framework**: LangChain Deep Agent with manual tool orchestration
- **Responsibilities**:
  - Receives code chunks and smell type to detect
  - Uses 4 specialized tools for analysis:
    - `analyze_code_structure()` - Extract metrics and patterns
    - `retrieve_similar_patterns()` - RAG context retrieval
    - `classify_severity_level()` - Determine severity rules
    - `extract_refactoring_suggestions()` - Generate improvements
  - Calls LLM with enriched context from tools
  - Parses LLM responses into structured findings
  - Assigns severity levels (LOW, HIGH, CRITICAL)
  - Generates explanations and refactoring suggestions
- **Module**: `src/code_smell_detector.py` (Enhanced Feb 26, 2026)
- **System Prompt**: "You are an expert software engineer specializing in detecting production code smells..."

**3. RAG Retriever (Custodian)**
- **Role**: Finds relevant examples from knowledge base
- **Responsibilities**:
  - Generates embeddings for input code
  - Performs similarity search in ChromaDB
  - Filters and ranks retrieved examples
  - Provides top-k most relevant examples to detector
  - Tracks retrieval metrics (relevance scores, latency)
- **Module**: `src/rag_retriever.py`
- **System Prompt**: "You are a knowledge base curator specializing in identifying relevant code smell examples for analysis."

**4. Quality Validator (CodeReviewer)**
- **Role**: Validates and reviews detection results
- **Responsibilities**:
  - Reviews detected smells for false positives
  - Cross-validates with static analysis tools (optional)
  - Assesses confidence scores
  - Provides quality ratings (1=requires revision, 2=needs improvement, 3=accepted)
  - Generates refactoring suggestions
- **Module**: `src/quality_validator.py`
- **System Prompt**: "You are a senior code reviewer evaluating the accuracy of code smell detections and providing actionable refactoring guidance."

### 🔄 Agent Workflow

```
User/Experiment
    ↓
Analysis Coordinator
    ↓
    ├─→ RAG Retriever ──→ (retrieve examples from ChromaDB)
    │                       ↓
    ├─→ Code Smell Detector ←─ (augmented with RAG context)
    │                       ↓
    └─→ Quality Validator ──→ (validate & review detections)
                            ↓
                     Final Results
```

### 📊 Agent Tracking (SQLite Database)

Similar to the reference architecture, we track all agent interactions:
- **agents** table: Agent metadata (name, role, system_prompt)
- **agent_requests** table: All LLM requests from agents
- **agent_responses** table: All LLM responses to agents
- **agent_actions** table: Actions performed by agents
- **analysis_runs** table: Links agents to specific analysis tasks

This enables:
- ✅ Reproducible experiments
- ✅ Token usage tracking
- ✅ Performance analysis per agent
- ✅ Visualization of agent interactions

---

## Research Objectives

### Primary Research Questions

This work breakdown structure is designed to systematically address four primary research questions:

**RQ1: How accurately do locally-deployed open-source language models detect code smells when evaluated against expert-validated ground truth compared to established static analysis tools?**

- **Addressed in**: Phase 3.3 (Initial Experiments), Phase 4.1 (Quantitative Evaluation)
- **Method**: Evaluate local LLMs (via Ollama) on MaRV dataset (95%+ accurate ground truth)
- **Comparison**: SonarQube, PMD, Checkstyle, SpotBugs, IntelliJ IDEA
- **Metrics**: Precision, Recall, F1-score per smell type and overall
- **Target**: 70-85% F1-score (competitive with cloud LLMs but privacy-preserving)

**RQ2: Does retrieval-augmented generation improve detection accuracy compared to vanilla prompting approaches?**

- **Addressed in**: Phase 3.3 (Baseline vs RAG Experiments), Phase 4.2 (Ablation Studies)
- **Method**: Compare vanilla LLM prompting vs. RAG-enhanced with ChromaDB retrieval
- **Hypothesis**: RAG provides +10-15% accuracy improvement, 20-30% false positive reduction
- **Ablation**: Test different top-k values (1, 3, 5, 10), embedding models, retrieval strategies
- **Evidence**: Per-smell accuracy improvement, example relevance analysis

**RQ3: Which specific code smell types exhibit the highest and lowest detection accuracy, and what factors influence these performance variations?**

- **Addressed in**: Phase 4.1 (Per-smell-type metrics), Phase 4.3 (Qualitative Analysis)
- **Method**: Break down performance by smell type (Long Method, God Class, Feature Envy, etc.)
- **Analysis**: Error categorization, pattern identification in failures
- **Factors**: Code complexity, language-specific features, smell ambiguity
- **Output**: Per-smell performance table, failure pattern taxonomy

**RQ4: What are the computational resource requirements and latency characteristics of local language model deployment for practical code review integration?**

- **Addressed in**: Phase 3.3 (Performance metrics), Phase 4.1 (Resource profiling)
- **Metrics**: 
  - Inference latency per code snippet
  - Memory consumption (RAM, GPU if used)
  - Token usage per analysis
  - Throughput (analyses per minute)
- **Comparison**: Local LLM vs. cloud API latency/cost tradeoffs
- **Practical Guidelines**: Hardware recommendations, optimization strategies

### Research Goals

**1. Empirical Contribution**
- Generate rigorous empirical evidence quantifying local LLM effectiveness for code smell detection
- Use expert-validated benchmarks (MaRV dataset with 95%+ accuracy)
- Systematic comparison against 5 established static analysis tools
- Per-smell-type performance breakdown across Java, Python, JavaScript

**2. Methodological Contribution**
- Demonstrate RAG enhancement for code smell detection (first study to combine local LLM + RAG for this task)
- Multi-agent architecture for collaborative code analysis
- Privacy-preserving approach (no code sent to external services)

**3. Practical Contribution**
- Viable alternative to commercial LLM APIs (GPT-4, Claude) for privacy-sensitive code analysis
- Open-source implementation enabling community adoption
- Documented tradeoffs: accuracy vs. cost vs. latency vs. privacy
- Deployment guidelines for CI/CD integration

### How WBS Phases Map to Research Objectives

| Phase | Primary RQ Addressed | Deliverable |
|-------|---------------------|-------------|
| Phase 1 (Weeks 1-2) | Literature foundation | Research gaps, baseline understanding |
| Phase 2 (Weeks 3-6) | Implementation infrastructure | Working multi-agent system, RAG pipeline |
| Phase 3 (Weeks 6-8) | RQ1, RQ2 setup | Datasets indexed, initial experiment results |
| Phase 4 (Weeks 8-10) | **RQ1, RQ2, RQ3, RQ4** | Complete evaluation, ablation studies, analysis |
| Phase 5 (Weeks 8-11) | All RQs | Research paper documenting findings |
| Phase 6-7 (Weeks 11-12) | RQ4 (deployment) | Optional demo tool, visualization |

### Expected Outcomes

Based on related work and preliminary analysis:

**Performance Targets (RQ1, RQ2):**
- Vanilla local LLM: 70-75% F1-score
- RAG-enhanced local LLM: **80-85% F1-score** (target)
- Baseline tools: 55-70% F1-score
- Commercial cloud LLMs: 85-90% F1-score (reference from literature)

**RAG Impact (RQ2):** (Architecture Section 3)
- Accuracy improvement: +10-15 percentage points
- False positive reduction: 20-30%
- Retrieval latency overhead: <200ms per query (50-100ms typical)
- Hallucination reduction: ~30-40% vs. vanilla prompting

**Per-Smell Performance (RQ3):**
- Best: Long Method, God Class (structural, metrics-based) - 85-90% F1
- Moderate: Feature Envy, Data Clumps (contextual) - 75-80% F1
- Challenging: Shotgun Surgery, Divergent Change (multi-file) - 60-70% F1

**Resource Requirements (RQ4):** (Architecture Section 1.1)
- Memory: **4-8GB RAM** (model-dependent, Llama 3 8B baseline)
- Latency: **2-5 seconds** per code snippet (function/method level)
- Throughput: **100-200 analyses/hour** (single CPU/GPU instance)
- GPU: Optional but recommended for faster inference
- Storage: ~1-2GB for ChromaDB vector store
- Cost: **$0** (vs. $0.01-0.10 per request for cloud APIs)

---

## Benchmarking Strategy

To ensure rigorous, reproducible evaluation of our local LLM approach, we establish comprehensive benchmarking protocols covering accuracy, performance, and resource utilization.

### 1. Datasets & Ground Truth

**Primary Benchmark: MaRV Dataset**
- **Source**: Karakoc et al. (2023) - Expert-validated code smell dataset
- **Accuracy**: 95%+ ground truth accuracy (validated by software engineering experts)
- **Languages**: Java, Python, JavaScript, C++
- **Smell Types**: 12+ production code smells
  - Bloaters: Long Method, God Class, Large Class, Long Parameter List
  - Couplers: Feature Envy, Inappropriate Intimacy, Message Chains
  - Change Preventers: Divergent Change, Shotgun Surgery
  - Dispensables: Duplicate Code, Lazy Class, Data Class, Dead Code
  - OO Abusers: Switch Statements, Refused Bequest
- **Dataset Size**: Target 500-1000 code samples per language
- **Annotation**: Binary labels (smell present/absent) + severity levels

**Secondary Benchmarks**
- **Qualitas Corpus**: Large-scale Java systems for real-world validation
- **PySmell**: Python-specific code smell instances
- **Custom Dataset**: 100+ manually verified examples per smell type for ablation studies

**Data Splits**
- **Training Set (RAG Knowledge Base)**: 60% - Index for retrieval
- **Validation Set**: 20% - Hyperparameter tuning (top-k, retrieval threshold)
- **Test Set**: 20% - Final evaluation (never seen during development)

### 2. Baseline Tool Configuration

To ensure fair comparison, all baseline tools are configured with:

**Tool Versions (Fixed)**
- SonarQube: v10.x Community Edition
- PMD: v7.x with built-in rulesets
- Checkstyle: v10.x with Google/Sun coding conventions
- SpotBugs: v4.x with standard detectors
- IntelliJ IDEA: 2024.x inspection engine

**Configuration Standardization**
- Default rule sets for each tool (no custom tuning)
- Maximum analysis depth/timeout per file: 60 seconds
- False positive threshold: Default sensitivity
- Output format: Normalized to JSON for comparison
- Execution environment: Same hardware as LLM evaluation

**Baseline Execution Protocol**
1. Clean analysis environment (clear caches)
2. Analyze identical code samples from test set
3. Record: detection results, confidence scores, analysis time
4. Normalize outputs to common schema: `{file, line, smell_type, severity, confidence, explanation}`

### 3. Evaluation Metrics

**Primary Metrics (RQ1, RQ2)**

**Accuracy Metrics** (Architecture Section 10.1, 11.2)
- **Precision**: TP / (TP + FP) - How many detected smells are correct?
- **Recall**: TP / (TP + FN) - How many true smells did we find?
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) - Harmonic mean
- **Accuracy**: (TP + TN) / Total - Overall correctness
- **False Positive Rate**: FP / (FP + TN) - Rate of incorrect detections
- **Confidence Distribution**: Histogram of confidence scores across predictions

**Per-Smell-Type Breakdown (RQ3)** (Architecture Section 10.1)
- Calculate Precision, Recall, F1 for each of 12+ smell types
- Track per-smell accuracy separately (as per llm_metrics in Architecture)
- Identify best/worst performing smell categories
- Error analysis: False positives, false negatives, confusion matrix
- Compare per-smell performance: Local LLM vs. each baseline tool

**RAG Effectiveness (RQ2)** (Architecture Section 3: RAG Pipeline)
- **ΔF1**: F1-score improvement (RAG vs. Vanilla) - Target: +10-15 percentage points
- **FP Reduction Rate**: (FP_vanilla - FP_rag) / FP_vanilla × 100% - Target: 20-30% reduction
- **Retrieval Quality**: 
  - Precision@k for retrieved examples (k=5 primary, per Architecture)
  - NDCG@k (Normalized Discounted Cumulative Gain)
  - Relevance scores distribution
  - MMR (Maximal Marginal Relevance) diversity metric
- **Ablation Metrics**: Performance vs. top-k values (k ∈ {1, 3, 5, 10}), default k=5

**Performance Metrics (RQ4)**

**Latency Benchmarks** (Aligned with LLM Architecture Section 1.1)
- **End-to-End Latency**: Total time from code submission to results
  - Target: **2-5 seconds** per code snippet (function/method level)
  - Breakdown: Embedding time + Retrieval time + LLM inference time + Parsing
- **Component Latency**:
  - Embedding generation: < 100ms per snippet
  - Vector search (ChromaDB): < 50ms per query
  - LLM inference: 1-3 seconds per snippet
  - Response parsing & validation: < 50ms
  - Total overhead (non-LLM): < 200ms

**Throughput Benchmarks** (Architecture Section 1.1)
- **Analyses per Hour**: Target **100-200 analyses/hour** (single GPU/CPU)
- **Analyses per Minute**: ~2-3 snippets/minute (accounting for 2-5s latency)
- **Batch Processing**: Throughput for 100 files with parallel processing
- **Concurrent Requests**: Performance degradation with parallel analyses (1, 2, 4, 8 workers)

**Resource Utilization** (Architecture Section 1.1)
- **CPU Usage**: Average % during analysis (expected 40-80%)
- **Memory Footprint**:
  - Base memory (model loaded): **4-8GB RAM** (model-dependent)
  - Peak memory (during inference): 8-12GB
  - ChromaDB index size: Record disk usage (expected ~500MB-2GB)
- **GPU Utilization**: If applicable (CUDA-enabled LLM), track GPU memory and compute %
- **Token Usage**:
  - Average tokens per prompt (input): 500-2000 tokens
  - Average tokens per response (output): 200-800 tokens
  - Total tokens per analysis run
  - Cost comparison: $0 (local) vs. $0.01-0.10 (cloud API)

**LLM-Specific Quality Metrics** (Architecture Section 10.1)

These metrics track LLM behavior and quality beyond traditional ML metrics:

- **Hallucination Rate** (Architecture Section 7.3, 10.1):
  - Percentage of detections referencing code elements not present in input
  - Line number mismatches (referenced line doesn't contain claimed smell)
  - Non-existent method/class references
  - Target: < 5% hallucination rate
  - Detection method: Automated validation against AST + manual review

- **Response Validation Metrics**:
  - **JSON Parse Success Rate**: % of LLM responses that parse correctly
  - **Schema Validation Rate**: % of parsed responses matching expected schema
  - **Retry Rate**: How often we need to retry due to malformed responses
  - Target: > 95% first-attempt success, < 3% retry rate

- **Cache Performance** (Architecture Section 9.1, 10.1):
  - **Cache Hit Rate**: % of identical code snippets served from cache
  - **Cache Miss Rate**: % requiring new LLM inference
  - **Cache Size**: Memory footprint of response cache
  - **Time Saved**: Latency reduction from cache hits (expected ~90% faster)
  - Target: 20-40% hit rate on typical workloads

- **Confidence Calibration**:
  - **Confidence vs. Accuracy Correlation**: How well confidence scores predict correctness
  - **Over-confidence Rate**: High confidence (>0.8) but incorrect predictions
  - **Under-confidence Rate**: Low confidence (<0.5) but correct predictions
  - **Confidence Distribution**: Histogram across all predictions
  - Target: Strong positive correlation (r > 0.6)

- **Model Selection Metrics** (Architecture Section 2):
  - **Model Usage Distribution**: % of analyses using each model (Llama 3 8B/13B, CodeLlama, etc.)
  - **Model Performance**: Accuracy breakdown by model type
  - **Model Selection Accuracy**: Was the right model selected for the task?

- **Error Analysis**:
  - **Error Types**: Categorize errors (timeout, parse failure, validation failure, hallucination)
  - **Error Recovery**: Success rate of retry and fallback strategies
  - **Error Rate by Smell Type**: Which smells cause most failures?

### 4. Experimental Conditions

**Hardware Specifications**
- **CPU**: Document processor model, cores, clock speed
- **RAM**: Minimum 16GB (record actual available)
- **GPU**: Optional (document model if used)
- **Storage**: SSD recommended for ChromaDB

**Software Environment**
- **OS**: macOS (primary), Linux (secondary validation)
- **Python**: 3.11+
- **Ollama**: Latest stable version
- **LLM Models Tested**:
  - CodeLlama 7B/13B
  - DeepSeek-Coder 6.7B/33B
  - Llama 3 8B
  - (Document exact model versions)

**Controlled Variables**
- Temperature: 0.1 (low for deterministic outputs)
- Top-p: 0.9
- Max tokens: 2048
- Random seed: Fixed for reproducibility
- Same datasets across all experiments
- Same evaluation scripts for all tools

### 5. Statistical Validation

**Significance Testing**
- **McNemar's Test**: Compare paired binary classifications (LLM vs. baselines)
- **Paired t-test**: Compare F1-scores across multiple runs
- **Confidence Intervals**: 95% CI for all reported metrics
- **Effect Size**: Cohen's d for practical significance

**Reproducibility Requirements** (Architecture Section 11.3)
- **Multiple Runs**: 3+ independent runs with different random seeds
- **Standard Deviation**: Report mean ± std for all metrics
- **Outlier Analysis**: Identify and document anomalous results
- **Version Control**: Lock all dependency versions (langchain, chromadb, langgraph, sentence-transformers)
- **Docker Image**: Create reproducible Docker container with all dependencies
- **Model Versions**: Document exact Ollama model versions (llama3:8b, codellama:7b, etc.)
- **Dataset Versioning**: Pin MaRV dataset version/commit hash

**Cross-Validation (for RAG hyperparameters)**
- 5-fold cross-validation for top-k selection
- Grid search: top-k ∈ {1, 3, 5, 10} (default k=5 per Architecture), threshold ∈ {0.5, 0.6, 0.7, 0.8}
- Embedding model comparison: all-MiniLM-L6-v2 (default) vs. alternatives
- MMR diversity parameter (λ ∈ {0.5, 0.7, 0.9})

### 6. Benchmark Execution Plan

**Phase 3.3 (Initial Experiments) - Week 7**
- Baseline tool execution on test set
- Vanilla LLM evaluation (no RAG)
- RAG-enhanced LLM evaluation
- Initial performance profiling

**Phase 4.1 (Quantitative Evaluation) - Week 8-9**
- Full test set evaluation (all tools + LLM variants)
- Per-smell-type metrics calculation
- Resource utilization benchmarking
- Statistical significance testing

**Phase 4.2 (Ablation Studies) - Week 9**
- RAG hyperparameter sweep (top-k, embedding models)
- Prompt engineering variations
- Model size comparison (7B vs. 13B vs. 33B)

**Phase 4.3 (Qualitative Analysis) - Week 10**
- Error analysis: Classify false positives/negatives
- Case studies: Best/worst detection examples
- Comparative analysis: Why LLM succeeds/fails vs. baselines

### 7. Results Documentation

**Benchmark Report Structure** (for paper Section 4: Results)

**Table 1: Overall Performance Comparison**
| Tool | Precision | Recall | F1-Score | Latency (s) | Cost |
|------|-----------|--------|----------|-------------|------|
| SonarQube | X.XX ± X.XX | X.XX ± X.XX | X.XX ± X.XX | X.X | Free |
| PMD | ... | ... | ... | ... | Free |
| Local LLM (Vanilla) | ... | ... | ... | ... | $0 |
| **Local LLM (RAG)** | **X.XX ± X.XX** | **X.XX ± X.XX** | **X.XX ± X.XX** | **X.X** | **$0** |
| GPT-4 (reference) | ... | ... | ... | X.X | $0.XX |

**Table 2: Per-Smell-Type Performance**
| Smell Type | Vanilla F1 | RAG F1 | ΔF1 | Best Baseline |
|------------|------------|--------|-----|---------------|
| Long Method | X.XX | X.XX | +X.XX | X.XX (Tool) |
| God Class | ... | ... | ... | ... |
| Feature Envy | ... | ... | ... | ... |

**Table 3: Resource Requirements**
| Metric | Vanilla | RAG | Overhead |
|--------|---------|-----|----------|
| Latency (s/snippet) | X.X ± X.X | X.X ± X.X | +X% |
| Memory (GB) | X.X | X.X | +X MB |
| Throughput (analyses/min) | XX | XX | -X% |

**Table 4: LLM Quality Metrics** (Architecture Section 10.1)
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Hallucination Rate (%) | < 5% | X.XX | ✅/❌ |
| Cache Hit Rate (%) | 20-40% | X.XX | ✅/❌ |
| JSON Parse Success (%) | > 95% | X.XX | ✅/❌ |
| Validation Failure (%) | < 3% | X.XX | ✅/❌ |
| Confidence Correlation (r) | > 0.6 | X.XX | ✅/❌ |

**Figures**
- Figure 1: F1-Score comparison (bar chart)
- Figure 2: Precision-Recall curves
- Figure 3: Per-smell heatmap
- Figure 4: Latency vs. Accuracy tradeoff
- Figure 5: RAG top-k ablation study
- **Figure 6**: Confidence calibration plot (Architecture Section 3)
- **Figure 7**: Hallucination rate over time (Architecture Section 7.3)
- **Figure 8**: Cache performance analysis (Architecture Section 9.1)

**Raw Data Preservation**
- All predictions saved to `results/predictions/{tool}_{date}.json`
- Confusion matrices: `results/confusion_matrices/`
- Timing logs: `results/performance/{tool}_timing.csv`
- Resource profiles: `results/resources/{tool}_profile.json`
- **Architecture-Specific Logs** (per Section 10.1):
  - `results/metrics/hallucination_rate.csv` - Hallucination detection logs
  - `results/metrics/cache_performance.csv` - Cache hit rate, miss rate
  - `results/metrics/validation_failures.csv` - LLM response validation failures
  - `results/metrics/confidence_scores.csv` - Confidence score distributions
  - `results/metrics/rag_retrieval_quality.csv` - Precision@k, NDCG@k per query

### 8. Benchmark Validity Threats

**Internal Validity**
- **Mitigation**: Fixed random seeds, multiple runs, controlled environment
- **Threat**: Model stochasticity → Report confidence intervals

**External Validity**
- **Mitigation**: Multiple datasets (MaRV + Qualitas + PySmell)
- **Threat**: Dataset bias → Document dataset characteristics

**Construct Validity**
- **Mitigation**: Use established metrics (P/R/F1), expert-validated ground truth
- **Threat**: Ground truth errors → Use 95%+ accurate MaRV dataset

**Conclusion Validity**
- **Mitigation**: Statistical significance testing, effect size reporting
- **Threat**: Cherry-picking results → Report all experiments, including failures

---

## Phase 1: Project Setup & Literature Review (Weeks 1-2)

### 1.1 Research Foundation
- [x] Research proposal document (8 sections complete)
- [x] Similar papers research (16 papers analyzed)
- [x] Research gap analysis (12 gaps identified)
- [x] Competitive positioning analysis (vs. NOIR, test smell papers)
- [x] Dataset comparison study (10 datasets evaluated)
- [x] LaTeX paper template setup (proposal.tex, references.bib)

### 1.2 Environment & Project Structure
- [x] Create simple project structure
  - [x] `src/` - Core Python modules
  - [x] `exp/` - Experiments directory (baseline, rag_experiments, ablation_studies)
  - [x] `data/` - Datasets (marv, qualitas_corpus, smelly_code, ground_truth)
  - [x] `results/` - Experiment results (see Benchmarking Strategy Section 7)
    - [x] `results/predictions/` - Tool predictions (baseline, llm_vanilla, llm_rag)
    - [x] `results/confusion_matrices/` - Confusion matrices
    - [x] `results/performance/` - Timing & latency logs
    - [x] `results/resources/` - Resource profiling data
    - [x] `results/figures/` - Generated plots for paper
    - [x] `results/tables/` - Generated tables (LaTeX, CSV)
    - [x] `results/metrics/` - LLM-specific metrics (hallucination_rate, cache_performance)
  - [x] `scripts/` - Utility scripts for benchmarking & experiments
  - [x] `visualization/` - Web-based visualization (optional)
  - [x] `paper/` - LaTeX source files
  - [x] `cache/` - LLM response cache directory (git-ignored)
  - [x] `chromadb_store/` - Vector database storage (git-ignored)
  - [x] `config.py` - Configuration file (Ollama, ChromaDB, RAG, M4 Pro optimizations)
  - [x] `requirements.txt` - Python dependencies (pinned versions)
- [x] Editor config (.editorconfig)
- [x] Cursor rules configuration
- [x] Git repository setup
- [x] Setup Python environment (conda/venv)
  - [x] Python 3.12.12 (venv created at project root)
  - [x] Core libraries: ollama, chromadb, sentence-transformers, tiktoken, langchain, langgraph, pydantic, structlog
- [x] Create `.gitignore` (exclude venv/, cache/, chromadb_store/, *.db, __pycache__)
- [x] Create `PROJECT_STRUCTURE.md` (directory organization and setup instructions)

---

## Phase 2: Core System Development (Weeks 3-6) ✅ COMPLETE (2.1, 2.2, 2.3, 2.4)

### 2.1 Configuration Module (Week 3: Feb 26 - Mar 1, 2026)

- [x] Create `config.py` at project root
  - [x] Ollama configuration (base URL: http://localhost:11434)
  - [x] Model selection (llama3:8b default, with model selection logic for fast/accurate/code-specialized)
  - [x] ChromaDB persistent directory path
  - [x] Embedding model name (sentence-transformers/all-MiniLM-L6-v2)
  - [x] Experiment result paths
  - [x] Dataset paths (MaRV, Qualitas Corpus, Smelly Code)
  - [x] Logging configuration
  - [x] M4 Pro optimizations (CPU inference, batch_size=5, max_concurrent_requests=2)
  - [x] Performance targets from Architecture (2-5s latency, 100-200 analyses/hour, 4-8GB RAM)
  - [x] RAG configuration (top_k=5, similarity_threshold=0.7, MMR diversity_lambda=0.7)

- [x] Create `requirements.txt` (Architecture Section 11.3 dependencies)
  - [x] ollama==0.3.3 (Python client)
  - [x] chromadb==0.5.5
  - [x] sentence-transformers==3.0.1
  - [x] tiktoken==0.7.0
  - [x] **langchain==0.2.16** - LLM orchestration framework
  - [x] **langgraph==0.2.28** - Workflow state machine (Architecture Section 6)
  - [x] pydantic==2.9.1 (validation)
  - [x] structlog==24.4.0 (logging)
  - [x] pygments==2.18.0 (code syntax)
  - [x] pandas==2.2.2, numpy==1.26.4 (for metrics)
  - [x] scikit-learn==1.5.1 (for metrics)
  - [x] matplotlib==3.9.2, seaborn==0.13.2 (plotting)
  - [x] pytest==8.3.3 (testing)
  - [x] flask==3.0.3 (optional - for visualization)

- [x] Create `Dockerfile` for reproducibility (Architecture Section 11.3)
  - [x] Base image: Python 3.14-slim (upgraded from 3.11)
  - [x] Install Ollama
  - [x] Copy requirements and install dependencies
  - [x] Copy source code
  - [x] Expose ports (if API needed)
  - [x] Document exact versions in image metadata
  - [x] Build and test Docker image locally (validated via scripts/validate_config.py)
  - [x] Create `.dockerignore` for optimized builds
  - [x] Create `docker-compose.yml` for multi-container setup

### 2.2 LLM Integration Module (Week 3-4: Feb 26 - Mar 7, 2026) ✅ COMPLETE

- [x] Create `src/llm_client.py`
  - [x] OllamaClient class for API communication
  - [x] Test connection to local Ollama (verify running)
  - [x] `generate()` method - single prompt completion
  - [x] `generate_stream()` method - streaming responses
  - [x] Handle timeouts and retries (exponential backoff)
  - [x] Token counting with tiktoken
  - [x] Response caching (file-based MD5 hash)

- [x] Create `src/prompt_templates.py`
  - [x] System prompt: Define code smell expert role with 17 smell types
  - [x] **Production code analysis prompts (Gap #12)**
  - [x] **Support for human-written & AI-generated code (Gap #11)**
  - [x] Few-shot examples for code smells:
    - [x] Long Method
    - [x] God Class
    - [x] Feature Envy
    - [x] (3 primary examples included)
  - [x] Structured JSON output format
  - [x] RAG context injection templates

- [x] Create `src/response_parser.py`
  - [x] Parse LLM JSON responses
  - [x] Extract code smell findings
  - [x] Validate severity levels (LOW, MEDIUM, HIGH, CRITICAL)
  - [x] Handle malformed responses with repair logic
  - [x] Confidence scoring (0.0-1.0)
  - [x] Type validation against 17 smell types

### 2.3 RAG Implementation (Week 4-5: Mar 8-21, 2026) ✅ COMPLETE

- [x] Create `src/embedding_service.py`
  - [x] Initialize HuggingFace embedding model (lazy loading for M4 Pro)
  - [x] Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
  - [x] `embed_text()` method with caching
  - [x] `embed_batch()` method for bulk processing (batch_size=32)
  - [x] Cache embeddings to disk (MD5 hash-based)
  - [x] Statistics tracking (cache hits/misses)

- [x] Create `src/vector_store.py`
  - [x] Initialize ChromaDB client with persistent storage
  - [x] Create collection for code smell examples
  - [x] `add_documents()` - index code examples
  - [x] `search()` - similarity search (top-k retrieval)
  - [x] `search_with_mmr()` - diversity-aware retrieval
  - [x] `get_stats()` - collection statistics
  - [x] Clear/reset collection

- [x] Create `src/rag_pipeline.py`
  - [x] `analyze_with_rag()` - RAG-enhanced code analysis
  - [x] `retrieve_context()` - get relevant examples from vector store
  - [x] `analyze_with_streaming()` - streaming RAG responses
  - [x] `initialize_knowledge_base()` - Knowledge base setup
  - [x] Context compression (respects token budget)
  - [x] Language detection (Python, Java, generic)

- [x] Create `src/code_chunker.py`
  - [x] AST-based code chunking (Python ast module)
  - [x] Support for Python (ast), Java (generic), and others
  - [x] Max chunk size: configurable (default 512 tokens)
  - [x] Overlap: configurable (default 50 tokens)
  - [x] Preserve code structure (function/class boundaries)
  - [x] Metadata tracking (decorators, names, types)

### 2.4 Workflow Engine & Agent Modules (Week 5: Mar 22-28, 2026) ✅ COMPLETE

**Multi-Agent System Implementation**

- [x] Create `src/analysis_coordinator.py` (Manager Agent) ✅
  - [x] Initialize Analysis Coordinator with system prompt
  - [x] `coordinate_analysis()` - Main orchestration method
  - [x] `split_code_into_chunks()` - Chunk large files
  - [x] `assign_detection_tasks()` - Distribute to detector agents
  - [x] `aggregate_results()` - Combine findings from all detectors
  - [x] `track_in_database()` - Log all agent actions
  - [x] Integration with database manager

- [x] Create `src/code_smell_detector.py` (Member Agent) ✅
  - [x] Initialize detector with specialization (e.g., "Long Method expert")
  - [x] `detect_smells()` - Main detection method
  - [x] `classify_and_assign_severity()` - LOW/HIGH/CRITICAL
  - [x] `generate_explanation()` - Why it's a code smell
  - [x] `suggest_refactoring()` - How to fix it
  - [x] Support for multiple programming languages
  - [x] Log all LLM interactions to database
  - [x] **Deep Agent Refactoring** (Feb 26 - Enhanced with LangChain)
    - [x] Implement LangChain Deep Agent pattern for short-duration code analysis
    - [x] `@tool analyze_code_structure()` - Extract metrics, functions, classes, complexity
    - [x] `@tool classify_severity_level()` - Determine CRITICAL/HIGH/MEDIUM/LOW with rules
    - [x] `@tool retrieve_similar_patterns()` - RAG retrieval of similar code patterns
    - [x] `@tool extract_refactoring_suggestions()` - Generate actionable improvements
    - [x] Manual tool orchestration (4-phase: structure → context → detection → classification)
    - [x] Statistics tracking (detection count, tool invocations, confidence, latency)
    - [x] Backward compatible with existing workflow and coordinator

- [x] Create `src/rag_retriever.py` (Custodian Agent) ✅
  - [x] Initialize RAG retriever with vector store
  - [x] `find_relevant_examples()` - Similarity search
  - [x] `rank_by_relevance()` - Score and sort examples
  - [x] `augment_context()` - Format examples for LLM prompt
  - [x] Cache retrieval results for efficiency
  - [x] Track retrieval metrics (precision@k)

- [x] Create `src/quality_validator.py` (CodeReviewer Agent) ✅
  - [x] Initialize validator with review criteria
  - [x] `validate_detection()` - Review a single finding
  - [x] `assign_confidence_score()` - 1-3 scale
  - [x] `cross_check_with_static_tools()` - Optional baseline comparison
  - [x] `suggest_improvements()` - Enhance detection quality
  - [x] Filter false positives

- [x] Create `src/code_analysis_workflow.py` ✅
  - [x] **Orchestration using agents** (not just sequential steps)
  - [x] Step 1: Coordinator receives code submission
  - [x] Step 2: Coordinator splits code into chunks
  - [x] Step 3: For each chunk:
    - [x] RAG Retriever finds relevant examples
    - [x] Detector analyzes code with RAG context
    - [x] Validator reviews detection results
  - [x] Step 4: Coordinator aggregates all results
  - [x] Step 5: Generate final report
  - [x] Error handling at each step
  - [x] Intermediate result logging

- [x] Create `src/common.py` (Utility functions) ✅
  - [x] `get_agent_name()` - Generate unique agent names from prompts
  - [x] `merge_results()` - Combine findings from multiple agents
  - [x] `to_safe_name()` - Sanitize names for file/folder storage
  - [x] Common helper functions

- [x] Create `src/code_parser.py` ✅
  - [x] Validate code syntax
  - [x] Detect programming language
  - [x] Parse AST (Python/Java)
  - [x] Extract code metrics (LOC, complexity)
  - [x] Identify code structures (classes, methods)

- [x] Create `src/logger.py` ✅
  - [x] Structured logging (JSON format)
  - [x] Log all agent activities
  - [x] Log all LLM requests/responses
  - [x] Log workflow steps
  - [x] Experiment tracking support

### 2.4.1 Deep Agent Enhancement & LangGraph Integration (Feb 26, 2026)

**LangGraph Workflow Integration with Deep Agents**

- [x] Create `src/workflow_graph.py` ✅ (520+ lines)
  - [x] LangGraph StateGraph for end-to-end code analysis workflow
  - [x] `AnalysisState` (Pydantic BaseModel) with 12 state fields
  - [x] Six workflow nodes:
    - [x] `parse_code_node()` - Language detection, validation, metrics extraction
    - [x] `chunk_code_node()` - AST-based code chunking
    - [x] `retrieve_context_node()` - RAG similarity search
    - [x] `detect_smells_node()` - Deep Agent LLM inference with tools
    - [x] `validate_findings_node()` - False positive filtering
    - [x] `aggregate_results_node()` - Result summary and compilation
  - [x] Linear workflow topology (ready for parallelization with Send())
  - [x] `WorkflowExecutor` class for async invocation
  - [x] Full integration with Deep Agent detectors

- [x] Refactor `src/code_smell_detector.py` to LangChain Deep Agent ✅ (420+ lines)
  - [x] Deep Agent pattern with 4 specialized tools
  - [x] Manual tool orchestration for Ollama compatibility
  - [x] 4-phase analysis: Structure → Context → Detection → Classification
  - [x] Tool invocation tracking and statistics
  - [x] Maintained backward compatibility

- [x] Documentation & Reference
  - [x] `LANGRAPH_LANGCHAIN_REFACTORING.md` - Design decisions and architecture
  - [x] `LANGGRAPH_LANGCHAIN_COMPLETION.md` - Detailed implementation report
  - [x] `PHASE_2_4_LANGGRAPH_SUMMARY.md` - Complete overview with diagrams
  - [x] `DEEP_AGENT_REFACTORING.md` - Deep Agent pattern documentation

**Testing & Validation**

- [x] Verify LangGraph workflow compilation
- [x] Test Deep Agent initialization with all 4 tools
- [x] Integration test with AnalysisCoordinator
- [x] Verify backward compatibility with existing modules

### 2.5 Database for Agent & Experiment Tracking ✅ COMPLETE (Mar 2, 2026)

**Database Schema for Multi-Agent System - Complete Implementation**

- [x] Create `src/database_manager.py` ✅ (850+ lines)
  - [x] SQLite database with thread-local connection pooling (M4 Pro optimized)
  - [x] **Agent-specific tables:**
    - [x] `agents` - Agent metadata (agent_id, name, role, system_prompt, framework, created_at, updated_at)
    - [x] `agent_requests` - All LLM requests from agents (request_id, agent_id, user_prompt, model_used, timestamp)
    - [x] `agent_responses` - All LLM responses (response_id, request_id, response_text, tokens_used, latency, timestamp)
    - [x] `agent_actions` - Actions performed by agents (action_id, agent_id, action_type, action_content, status, timestamp)
    - [x] `processes` - Workflow steps (process_id, process_type, agent_id, task_id, duration, timestamp)
  - [x] **Experiment tracking tables:**
    - [x] `experiments` - Experiment metadata (exp_id, name, config JSON, status, created_at, completed_at)
    - [x] `analysis_runs` - Each code analysis run (run_id, exp_id, code_snippet, language, result, created_at)
    - [x] `code_smell_findings` - Detected smells (finding_id, run_id, smell_type, severity, confidence, agent_id, explanation)
    - [x] `ground_truth` - Labeled data for evaluation (gt_id, code_snippet, smell_labels JSON, source, language, created_at)
  - [x] **CRUD operations:**
    - [x] `add_agent()` - Register new agent
    - [x] `add_request()` - Log LLM request
    - [x] `add_response()` - Log LLM response
    - [x] `add_action()` - Log agent action
    - [x] `add_process()` - Log workflow step
    - [x] `add_analysis_run()` - Log analysis execution
    - [x] `add_finding()` - Log code smell detection
    - [x] `get_agent_stats()` - Get agent performance metrics (requests, tokens, latency, actions, success rate, detections)
    - [x] `export_results()` - Export to CSV/JSON for analysis (runs, findings, agents)
  - [x] Database cleanup utilities (`cleanup()`, `vacuum()`)
  - [x] Database statistics and monitoring (`get_database_stats()`)
  - [x] Singleton pattern for resource efficiency

**Data Models (Pydantic)**
- [x] `Agent` - Agent metadata model
- [x] `AgentRequest` - LLM request model
- [x] `AgentResponse` - LLM response model
- [x] `AgentAction` - Agent action model
- [x] `CodeSmellFinding` - Detection result model
- [x] `SeverityLevel` - Enum for severity levels

**Testing & Validation**
- [x] All 9 tables creation verified
- [x] CRUD operations tested with sample data
- [x] Agent statistics calculation verified
- [x] Export to CSV/JSON tested
- [x] Connection pooling tested
- [x] Indices created for query performance
- [x] M4 Pro optimization: connection pooling, efficient queries
- [x] Integrated into Makefile: `make run-database`
- [x] All integration tests pass with `make run-all`

### 2.6 Benchmarking Infrastructure (Week 6: Mar 22-28, 2026)

**Setup tools and scripts for systematic benchmarking - See Benchmarking Strategy**

- [ ] Create `scripts/setup_baseline_tools.sh`
  - [ ] Install SonarQube, PMD, Checkstyle, SpotBugs
  - [ ] Configure IntelliJ IDEA inspection engine
  - [ ] Version documentation (lock versions per Benchmarking Section 2)
  - [ ] Test installations with sample code

- [ ] Create `scripts/run_baseline_tools.py`
  - [ ] Execute all 5 baseline tools on a code sample
  - [ ] Parse tool-specific output formats
  - [ ] Normalize to common JSON schema:
    ```json
    {
      "tool": "SonarQube",
      "file": "Example.java",
      "line": 42,
      "smell_type": "Long Method",
      "severity": "HIGH",
      "confidence": 0.85,
      "explanation": "..."
    }
    ```
  - [ ] Save to `results/predictions/baseline/{tool}.json`
  - [ ] Handle tool errors gracefully

- [ ] Create `src/benchmark_utils.py`
  - [ ] `calculate_metrics()` - Precision, Recall, F1, Accuracy
  - [ ] `confusion_matrix()` - Generate confusion matrix
  - [ ] `statistical_tests()` - McNemar's, t-test, confidence intervals
  - [ ] `per_smell_breakdown()` - Metrics for each smell type
  - [ ] `latency_profiler()` - Track component latency (embedding, retrieval, inference)
  - [ ] `resource_monitor()` - CPU, memory, GPU tracking during analysis
  - [ ] **Architecture-specific metrics (Section 10.1)**:
    - [ ] `calculate_hallucination_rate()` - Detect and measure hallucinations
    - [ ] `cache_hit_rate()` - Track cache performance
    - [ ] `validation_failure_rate()` - JSON parse/validation failures
    - [ ] `confidence_calibration()` - Confidence vs. accuracy correlation
    - [ ] `rag_retrieval_quality()` - Precision@k, NDCG@k calculation

- [ ] Create `src/result_exporter.py`
  - [ ] Export to LaTeX tables (for paper)
  - [ ] Export to CSV (for spreadsheet analysis)
  - [ ] Generate matplotlib/seaborn plots
  - [ ] Save figures to `paper/figures/`

- [ ] Create `results/` directory structure
  - [ ] `results/predictions/` - Tool predictions
  - [ ] `results/confusion_matrices/` - Confusion matrix data
  - [ ] `results/performance/` - Timing logs
  - [ ] `results/resources/` - Resource profiles
  - [ ] `results/figures/` - Generated plots
  - [ ] `results/tables/` - Generated tables

---

## Phase 3: Dataset & Experiments (Weeks 6-8)

### 3.1 Dataset Acquisition & Preprocessing (Week 6: Mar 29 - Apr 4, 2026)

**Focus: Production Code Smells Only (Gap #12)**

- [ ] Create `data/datasets/` directory structure
  - [ ] `data/datasets/marv/` - MaRV dataset
  - [ ] `data/datasets/qualitas_corpus/` - Java projects
  - [ ] `data/datasets/smelly_code/` - Labeled smell examples
  - [ ] `data/ground_truth/` - Manually verified examples

- [ ] Download MaRV dataset
  - [ ] Research dataset source
  - [ ] Download and extract
  - [ ] **Verify production code smell labels (not test smells)**
  - [ ] Document dataset structure in `data/datasets/marv/README.md`

- [ ] Download additional datasets
  - [ ] Qualitas Corpus (Java code smells)
  - [ ] Code Smell Detection datasets from literature
  - [ ] **Filter for production code only**

- [ ] Create `src/data_loader.py`
  - [ ] Load datasets from various formats (JSON, CSV, XML)
  - [ ] Parse code files
  - [ ] Extract smell annotations
  - [ ] Unified data model for all datasets

- [ ] Create `src/data_preprocessor.py`
  - [ ] Clean code samples (remove comments, normalize formatting)
  - [ ] Extract features (metrics)
  - [ ] Label validation (ensure production code smells)
  - [ ] **Train/validation/test split: 60/20/20** (per Benchmarking Section 1)
    - [ ] Training set (60%): For RAG knowledge base indexing
    - [ ] Validation set (20%): For hyperparameter tuning
    - [ ] Test set (20%): For final evaluation (never seen during development)
  - [ ] Save preprocessed data with split labels

### 3.2 Vector Store Indexing (Week 6: Apr 1-4, 2026)

- [ ] Create `scripts/index_datasets.py`
  - [ ] Load preprocessed datasets
  - [ ] Generate embeddings for all code examples
  - [ ] Batch processing for efficiency
  - [ ] Index into ChromaDB
  - [ ] Track indexing progress
  - [ ] Save indexing statistics

- [ ] Test retrieval quality
  - [ ] Sample queries for each smell type
  - [ ] Evaluate top-k accuracy
  - [ ] Adjust embedding model if needed

### 3.3 Initial Experiments (Week 7-8: Apr 5-18, 2026)

**Benchmark Execution: See "Benchmarking Strategy" section for detailed protocols**

- [ ] Setup baseline tools (per Benchmarking Strategy Section 2)
  - [ ] Install SonarQube v10.x, PMD v7.x, Checkstyle v10.x, SpotBugs v4.x
  - [ ] Configure IntelliJ IDEA 2024.x inspection engine
  - [ ] Create `scripts/run_baseline_tools.py` - Execute all 5 baseline tools
  - [ ] Normalize outputs to common JSON schema
  - [ ] Save baseline predictions to `results/predictions/baseline/`

- [ ] Create `exp/baseline/` directory
  - [ ] Experiment on baseline LLM (no RAG)
  - [ ] Compare different LLM models (CodeLlama 7B/13B, DeepSeek-Coder 6.7B/33B)
  - [ ] Test prompt variations
  - [ ] Fixed experimental conditions: temp=0.1, top_p=0.9, seed=42

- [ ] Create `exp/rag_experiments/` directory
  - [ ] Experiment with RAG-enhanced detection
  - [ ] Vary top-k retrieval (k = 1, 3, 5, 10) - See Benchmarking Section 5
  - [ ] Test different embedding models (all-MiniLM-L6-v2, bge-small-en-v1.5)
  - [ ] Compare retrieval strategies

- [ ] Create `scripts/run_experiment.py`
  - [ ] Main experiment runner
  - [ ] Configuration via command-line args or config file
  - [ ] Batch processing of test set (20% holdout from Benchmarking Section 1)
  - [ ] Save results to database and `results/predictions/{experiment}/`
  - [ ] Progress tracking with ETA
  - [ ] **Resource profiling**: Track CPU, memory, latency per analysis

- [ ] Performance metrics calculation (See Benchmarking Section 3)
  - [ ] Precision, Recall, F1-score per smell type (12+ smells)
  - [ ] Overall accuracy
  - [ ] **Comparison with existing tools (Gap #7)**
  - [ ] **Performance benchmarks (Benchmarking Section 3)**:
    - [ ] Latency measurements (embedding + retrieval + inference)
    - [ ] Throughput (analyses per hour, target 100-200/hr)
    - [ ] Memory footprint (base + peak, target 4-8GB)
    - [ ] Token usage per analysis
  - [ ] **LLM-Specific Quality Metrics (Architecture Section 10.1)**:
    - [ ] Hallucination rate tracking (target < 5%)
    - [ ] Cache hit rate (target 20-40%)
    - [ ] JSON parse success rate (target > 95%)
    - [ ] Validation failure rate (target < 3%)
    - [ ] Confidence score distribution
    - [ ] Model selection distribution
  - [ ] Generate initial results tables (Benchmarking Section 7, Table 1 draft)

---

## Phase 4: Evaluation & Analysis (Weeks 8-10)

### 4.1 Quantitative Evaluation (Week 8-9: Apr 12-25, 2026)

**Full Benchmark Execution: See "Benchmarking Strategy" Section 6**

- [ ] Create `src/evaluation.py`
  - [ ] Load ground truth data (MaRV dataset test set - 20% split)
  - [ ] Load prediction results from all tools/experiments
  - [ ] Calculate metrics (Benchmarking Section 3):
    - [ ] Precision, Recall, F1 (per smell type & overall)
    - [ ] True Positive Rate, False Positive Rate
    - [ ] Confusion matrix (12+ smell types × 6+ tools)
    - [ ] ROC curves (if applicable)
  - [ ] **Statistical significance tests (Benchmarking Section 5)**:
    - [ ] McNemar's test (LLM vs. each baseline)
    - [ ] Paired t-test for F1-scores
    - [ ] 95% confidence intervals
    - [ ] Cohen's d effect size
  - [ ] **Compare with baselines from literature (Gap #7)**

- [ ] Create `scripts/analyze_results.py`
  - [ ] Aggregate results across 3+ independent runs (different seeds)
  - [ ] Calculate mean ± standard deviation for all metrics
  - [ ] **Generate benchmark report tables (Benchmarking Section 7)**:
    - [ ] Table 1: Overall Performance Comparison (6+ tools)
    - [ ] Table 2: Per-Smell-Type Performance (12+ rows)
    - [ ] Table 3: Resource Requirements (latency, memory, throughput)
  - [ ] Plot performance graphs:
    - [ ] Figure 1: F1-Score comparison (bar chart)
    - [ ] Figure 2: Precision-Recall curves
    - [ ] Figure 3: Per-smell heatmap
    - [ ] Figure 4: Latency vs. Accuracy tradeoff
  - [ ] Identify best configurations (model, top-k, embedding)
  - [ ] Export to LaTeX tables for paper

- [ ] Error analysis
  - [ ] Categorize false positives and false negatives
  - [ ] Identify patterns in failures (code complexity, smell ambiguity)
  - [ ] **Analyze AI-generated vs human-written code (Gap #11)**
  - [ ] Create failure taxonomy
  - [ ] Select case study examples for qualitative analysis

- [ ] Resource profiling (RQ4 - Benchmarking Section 3)
  - [ ] Analyze performance logs from `results/performance/`
  - [ ] Calculate average latency components (embedding, retrieval, inference)
  - [ ] Peak memory usage analysis
  - [ ] Throughput under load testing
  - [ ] Cost comparison (local $0 vs. cloud API ~$0.05/request)

### 4.2 Ablation Studies (Week 9: Apr 19-25, 2026)

**Systematic Ablation: See Benchmarking Strategy Section 5 (Cross-Validation)**

- [ ] Ablation: RAG vs No RAG (RQ2)
  - [ ] Measure ΔF1, false positive reduction rate
  - [ ] Calculate retrieval quality (Precision@k, NDCG@k)
  - [ ] Generate Figure 5: RAG top-k ablation study

- [ ] Ablation: Top-k retrieval values (k ∈ {1, 3, 5, 10})
  - [ ] 5-fold cross-validation on validation set
  - [ ] Identify optimal k value
  - [ ] Plot F1 vs. k curve

- [ ] Ablation: Different embedding models
  - [ ] all-MiniLM-L6-v2 (384-dim, lightweight)
  - [ ] bge-small-en-v1.5 (384-dim, better quality)
  - [ ] Compare retrieval accuracy and latency

- [ ] Ablation: Different LLM models
  - [ ] CodeLlama 7B vs. 13B (accuracy vs. latency tradeoff)
  - [ ] DeepSeek-Coder 6.7B vs. 33B
  - [ ] Llama 3 8B
  - [ ] Compare per-smell performance

- [ ] Ablation: Few-shot examples count (0, 1, 3, 5)
- [ ] Ablation: Temperature settings (0.0, 0.1, 0.3)
- [ ] **Ablation: Privacy-preserving (local) vs cloud LLMs (Gap #1)**
  - [ ] Reference GPT-4 performance from literature
  - [ ] Compare local F1 vs. cloud F1, latency, cost, privacy

- [ ] Document findings in `exp/ablation_studies/README.md`
  - [ ] Include all ablation result tables
  - [ ] Best configuration recommendations
  - [ ] Tradeoff analysis (accuracy vs. resources)

### 4.3 Qualitative Analysis (Week 9-10: Apr 19 - May 2, 2026)

- [ ] Case studies for each smell type
  - [ ] Select representative examples
  - [ ] Analyze LLM reasoning
  - [ ] Evaluate explanation quality
  - [ ] Assess refactoring suggestions

- [ ] User feedback (if time permits)
  - [ ] Demo to developers
  - [ ] Gather feedback on usefulness
  - [ ] Document insights

---

## Phase 5: Paper Writing (Weeks 8-11, Parallel with Evaluation)

### 5.1 Paper Structure & Writing (Weeks 8-10: Apr 12 - May 2, 2026)

- [ ] Setup LaTeX environment in `paper/`
  - [ ] Copy from `docs/research/proposal-latex/`
  - [ ] Paper template (IEEE/ACM/EMNLP format - choose based on target venue)
  - [ ] `paper/main.tex` - Main paper file
  - [ ] `paper/references.bib` - Bibliography
  - [ ] `paper/figures/` - Figures and plots
  - [ ] `paper/tables/` - Result tables

- [ ] Abstract (1 page)
  - [ ] Problem statement
  - [ ] **Novelty: Privacy-preserving, RAG-enhanced, local LLM (Gaps #1, #4, #5)**
  - [ ] **Production code focus (Gap #12)**
  - [ ] Key results
  - [ ] Contributions

- [ ] Introduction (2 pages)
  - [ ] Motivation: Code smell detection challenges
  - [ ] **Research gaps from literature (12 gaps)**
  - [ ] Proposed solution overview
  - [ ] **Contributions: Privacy, RAG, local LLM, production code**
  - [ ] Paper organization

- [ ] Related Work (2-3 pages)
  - [ ] Traditional code smell detection
  - [ ] ML-based approaches
  - [ ] **LLM-based approaches (recent, Gaps #2, #3)**
  - [ ] **RAG in code analysis (Gap #4, #5)**
  - [ ] **Privacy in code analysis (Gap #1)**
  - [ ] **Test smell vs production smell (Gap #12)**
  - [ ] Comparison table with existing work

- [ ] Methodology (3-4 pages)
  - [ ] System architecture diagram
  - [ ] LLM selection (Ollama, local models)
  - [ ] RAG pipeline (embeddings, ChromaDB, retrieval)
  - [ ] Prompt engineering strategy
  - [ ] Workflow design
  - [ ] **Privacy considerations**

- [ ] Experimental Setup (2 pages)
  - [ ] Datasets description (MaRV, others)
  - [ ] Evaluation metrics
  - [ ] Baselines
  - [ ] Implementation details
  - [ ] Hardware/software environment

- [ ] Results & Discussion (3-4 pages)
  - [ ] **Import benchmark tables from Benchmarking Strategy Section 7**:
    - [ ] Table 1: Overall Performance Comparison (all tools + LLM variants)
    - [ ] Table 2: Per-Smell-Type Performance (12+ smells)
    - [ ] Table 3: Resource Requirements (latency, memory, throughput)
  - [ ] **Import benchmark figures**:
    - [ ] Figure 1: F1-Score comparison (bar chart)
    - [ ] Figure 2: Precision-Recall curves
    - [ ] Figure 3: Per-smell heatmap
    - [ ] Figure 4: Latency vs. Accuracy tradeoff
    - [ ] Figure 5: RAG top-k ablation study
  - [ ] **Answer RQ1**: Detection accuracy vs. baselines
    - [ ] Report F1-scores with 95% CI
    - [ ] McNemar's test results (statistical significance)
    - [ ] Best/worst performing smell types
  - [ ] **Answer RQ2**: RAG effectiveness (Gap #4, #5)
    - [ ] ΔF1 improvement, false positive reduction rate
    - [ ] Retrieval quality metrics (Precision@k)
    - [ ] Optimal top-k value from ablation
  - [ ] **Answer RQ3**: Per-smell analysis
    - [ ] Why Long Method/God Class perform well (85-90% F1)
    - [ ] Why Shotgun Surgery/Divergent Change are challenging (60-70% F1)
    - [ ] Failure pattern taxonomy
  - [ ] **Answer RQ4**: Resource requirements
    - [ ] Latency breakdown (embedding + retrieval + inference)
    - [ ] Memory footprint, throughput
    - [ ] Local vs cloud LLM trade-offs (Gap #1): cost, privacy, latency
  - [ ] Error analysis insights
  - [ ] Discussion of findings and implications

- [ ] Threats to Validity (1 page)
  - [ ] Internal validity
  - [ ] External validity
  - [ ] Construct validity
  - [ ] Mitigation strategies

- [ ] Conclusion & Future Work (1 page)
  - [ ] Summary of contributions
  - [ ] Key findings
  - [ ] Limitations
  - [ ] Future research directions

### 5.2 Figures & Tables (Week 10: Apr 26 - May 2, 2026)

**Generate from Benchmarking Results - See Benchmarking Strategy Section 7**

- [ ] Create figures (from `scripts/analyze_results.py` outputs)
  - [ ] **Figure 1**: F1-Score comparison bar chart (6+ tools)
  - [ ] **Figure 2**: Precision-Recall curves (all approaches)
  - [ ] **Figure 3**: Per-smell performance heatmap (12+ smells × 6+ tools)
  - [ ] **Figure 4**: Latency vs. Accuracy tradeoff scatter plot
  - [ ] **Figure 5**: RAG top-k ablation study (F1 vs. k)
  - [ ] System architecture diagram (methodology section)
  - [ ] RAG pipeline flowchart (methodology section)
  - [ ] Example code smell detection (annotated walkthrough)

- [ ] Create tables (from `results/` benchmark outputs)
  - [ ] **Table 1**: Overall Performance Comparison
    - [ ] Columns: Tool | Precision | Recall | F1-Score | Latency | Cost
    - [ ] Rows: 5 baselines + 2-3 LLM variants + GPT-4 reference
    - [ ] Include mean ± std from 3+ runs
  - [ ] **Table 2**: Per-Smell-Type Performance
    - [ ] Columns: Smell Type | Vanilla F1 | RAG F1 | ΔF1 | Best Baseline
    - [ ] 12+ smell types with category groupings
  - [ ] **Table 3**: Resource Requirements
    - [ ] Columns: Metric | Vanilla | RAG | Overhead
    - [ ] Rows: Latency, memory, throughput, token usage
  - [ ] Dataset statistics (MaRV breakdown by language/smell)
  - [ ] Ablation study results (embedding models, LLM models, top-k)
  - [ ] Comparison with related work (our approach vs. literature)
  - [ ] Code smell taxonomy (12+ smells with definitions)

### 5.3 Paper Refinement (Week 11: May 3-9, 2026)

- [ ] First complete draft
- [ ] Internal review (self-review, check gaps)
- [ ] **Verify all 12 research gaps are addressed**
- [ ] Revisions based on feedback
- [ ] Proofreading and formatting
- [ ] Check references (all 16+ papers cited)
- [ ] **Reproducibility Package** (Architecture Section 11.3)
  - [ ] Code availability statement (GitHub repository link)
  - [ ] **Docker image** with all dependencies (Ollama, ChromaDB, etc.)
    - [ ] Dockerfile with exact versions
    - [ ] Docker Compose for easy deployment
    - [ ] Pre-built image pushed to DockerHub (optional)
  - [ ] **Installation guide** (`docs/deployment/DEPLOYMENT_GUIDE.md`)
    - [ ] System requirements (4-8GB RAM, CPU/GPU specs)
    - [ ] Step-by-step installation instructions
    - [ ] Troubleshooting common issues
  - [ ] **Experiment reproduction instructions**
    - [ ] Scripts to download MaRV dataset
    - [ ] Commands to reproduce all experiments
    - [ ] Expected output formats
  - [ ] **Model versions documentation**
    - [ ] Exact Ollama model versions (llama3:8b, codellama:7b)
    - [ ] Embedding model version (all-MiniLM-L6-v2)
    - [ ] Dependency versions (requirements.txt with pinned versions)
  - [ ] Dataset description and access instructions
  - [ ] Additional experimental results (full confusion matrices, per-sample predictions)
  - [ ] **Replication command**: Single command to run all experiments
    - [ ] Example: `./scripts/replicate_study.sh`

---

## Phase 6: Optional System Deployment (Week 11-12, If Time Permits)

### 6.1 Simple API Wrapper (Week 11: May 3-9, 2026)

**Note**: This is for practical demonstration, not required for paper.

- [ ] Create `src/api_server.py` (optional FastAPI app)
  - [ ] Simple Flask or FastAPI wrapper
  - [ ] Endpoint: `/analyze` - analyze code
  - [ ] Endpoint: `/health` - health check
  - [ ] No authentication (research demo only)

- [ ] Create `src/cli.py` (command-line interface)
  - [ ] Accept code file or stdin
  - [ ] Display detected smells
  - [ ] Save results to file

### 6.2 Visualization (Week 11-12: May 3-16, 2026)

- [ ] Create `visualization/` directory (simple Flask app)
  - [ ] Web interface to display analysis results
  - [ ] **Multi-agent interaction visualization** (similar to magis)
  - [ ] Show detected smells with explanations
  - [ ] Visualize confidence scores
  - [ ] Compare different experiments
  - [ ] Timeline view of agent activities

- [ ] Create `visualization/process.py` (Flask server)
  - [ ] Connect to SQLite database
  - [ ] Query agent activities, requests, responses
  - [ ] Serve agent interaction timeline
  - [ ] Export analysis results

- [ ] Create `visualization/db_config.py`
  - [ ] Database connection settings
  - [ ] Query helpers for agent data

- [ ] Create `visualization/templates/`
  - [ ] `index.html` - Main dashboard
  - [ ] `analysis.html` - Analysis results view
  - [ ] `experiments.html` - Experiment comparison
  - [ ] `agents.html` - **Agent interaction timeline**
  - [ ] `role_icon.html` - **Agent role icons** (Manager, Detector, Retriever, Validator)

- [ ] Create `visualization/static/`
  - [ ] `index.css` - Main styles
  - [ ] `task.css` - Task/analysis view styles
  - [ ] `task.js` - Interactive timeline
  - [ ] `agents.js` - **Agent interaction visualization**

---

## Phase 7: Final Deliverables (Week 12: May 10-16, 2026)

### 7.1 Documentation

- [ ] Update `README.md`
  - [ ] Project overview
  - [ ] Research goals & novelty
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Dataset information
  - [ ] Experiment reproduction steps
  - [ ] Citation information (arXiv link)

- [ ] Create `EXPERIMENTS.md`
  - [ ] How to run experiments
  - [ ] Configuration options
  - [ ] Expected results
  - [ ] Troubleshooting

- [ ] Create `ARCHITECTURE.md`
  - [ ] System design overview
  - [ ] Module descriptions
  - [ ] Data flow diagrams

### 7.2 Code Quality

- [ ] Code cleanup and refactoring
- [ ] Add docstrings to all modules
- [ ] Type hints (Python typing)
- [ ] Unit tests for core modules (if time)
- [ ] Code formatting (black, isort)

### 7.3 Research Artifacts

- [ ] Create GitHub release
- [ ] Share datasets (if allowed)
- [ ] Publish experiment results
- [ ] Create Zenodo archive (DOI)

---

## Summary: Research-Oriented Approach

### Key Principles

✅ **Simple Python modules** - Core scripts like the example codebase  
✅ **Research-first** - Focus on paper publication and experiments  
✅ **Dual-track** - Research (paper) + Implementation (working system)  
✅ **Local-first** - Ollama running locally, privacy-preserving  
✅ **Production code** - Not test smells (Gap #12)  

### Project Structure (Simple)

```
code-smell/
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── README.md                 # Main documentation
├── EXPERIMENTS.md            # How to run experiments
├── ARCHITECTURE.md           # System design
│
├── src/                      # Core Python modules
│   ├── llm_client.py                  # Ollama client wrapper
│   ├── prompt_templates.py            # LLM prompts for detection
│   ├── response_parser.py             # Parse LLM responses
│   ├── embedding_service.py           # Text embeddings
│   ├── vector_store.py                # ChromaDB interface
│   ├── rag_pipeline.py                # RAG implementation
│   ├── code_chunker.py                # AST-based chunking
│   ├── code_parser.py                 # Code parsing & metrics
│   ├── logger.py                      # Structured logging
│   ├── common.py                      # Utility functions
│   │
│   ├── # === Multi-Agent System ===
│   ├── analysis_coordinator.py        # Manager agent
│   ├── code_smell_detector.py         # Member agent (detector)
│   ├── rag_retriever.py               # Custodian agent (retriever)
│   ├── quality_validator.py           # CodeReviewer agent
│   ├── code_analysis_workflow.py      # Orchestrates agents
│   │
│   ├── database_manager.py            # SQLite tracking (agents + experiments)
│   ├── data_loader.py                 # Dataset loaders
│   ├── data_preprocessor.py           # Data preprocessing
│   ├── evaluation.py                  # Metrics & evaluation
│   ├── api_server.py                  # Optional FastAPI
│   └── cli.py                         # Command-line interface
│
├── exp/                      # Experiments
│   ├── baseline/                      # No RAG experiments
│   ├── rag_experiments/               # RAG-enhanced experiments
│   └── ablation_studies/              # Ablation tests
│
├── data/                     # Datasets
│   ├── datasets/
│   │   ├── marv/
│   │   ├── qualitas_corpus/
│   │   └── smelly_code/
│   └── ground_truth/
│
├── results/                  # Experiment results
│   ├── metrics/
│   ├── figures/
│   └── tables/
│
├── scripts/                  # Helper scripts
│   ├── index_datasets.py
│   ├── run_experiment.py
│   └── analyze_results.py
│
├── visualization/            # Optional web UI (multi-agent timeline)
│   ├── process.py                     # Flask server
│   ├── db_config.py                   # Database connection
│   ├── static/
│   │   ├── index.css
│   │   ├── task.css
│   │   ├── task.js                    # Interactive timeline
│   │   └── agents.js                  # Agent visualization
│   └── templates/
│       ├── index.html                 # Main dashboard
│       ├── analysis.html              # Analysis results
│       ├── experiments.html           # Experiment comparison
│       ├── agents.html                # Agent interaction timeline
│       └── role_icon.html             # Agent role icons
│
├── paper/                    # LaTeX paper
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   └── tables/
│
└── docs/                     # Documentation (existing)
    ├── architecture/
    ├── planning/
    └── research/
```

### Timeline Summary

- **Weeks 1-2**: Setup & literature ✅
- **Weeks 3-6**: Core system development (simple Python modules)
- **Weeks 6-8**: Datasets & initial experiments
- **Weeks 8-10**: Evaluation & analysis
- **Weeks 8-11**: Paper writing (parallel)
- **Weeks 11-12**: Optional deployment & finalization

### Research Goals Addressed

| Gap | Description | How Addressed |
|-----|-------------|---------------|
| #1 | Privacy | Local LLM (Ollama), no cloud |
| #2-#3 | Limited LLM studies | RAG-enhanced LLM approach |
| #4-#5 | No RAG | ChromaDB + sentence-transformers |
| #6 | Limited validation | MaRV dataset (95%+ accurate) |
| #7 | No baselines | Compare 5 tools |
| #11 | AI-generated code | Support both human & AI code |
| #12 | Test smells focus | **Production code smells** |

### Multi-Agent System Summary

Our system employs **4 specialized agents** following a collaborative architecture:

| Agent | Role | Module | Key Responsibilities |
|-------|------|--------|---------------------|
| **Analysis Coordinator** | Manager | `analysis_coordinator.py` | Orchestrate workflow, split code, aggregate results |
| **Code Smell Detector** | Member | `code_smell_detector.py` | Detect smells, classify severity, generate explanations |
| **RAG Retriever** | Custodian | `rag_retriever.py` | Find relevant examples, rank by similarity, augment context |
| **Quality Validator** | CodeReviewer | `quality_validator.py` | Validate detections, filter false positives, suggest refactoring |

**Agent Interaction Flow:**
```
Analysis Coordinator (Manager)
    ↓ (assigns tasks)
    ├─→ RAG Retriever (finds examples) ────┐
    │                                       ↓
    ├─→ Code Smell Detector (analyzes) ←───┘ (augmented with context)
    │                                       ↓
    └─→ Quality Validator (reviews) ────→ Final Results
```

**Database Tracking:**
- All agent interactions logged to SQLite
- Enables reproducible experiments
- Tracks: requests, responses, actions, processes
- Supports visualization of agent timeline

---

**Note**: Optional FastAPI deployment is separated from core research. Focus is on working system + publishable paper, not enterprise-grade API.

**Last Updated**: February 26, 2026  
**Next Review**: March 5, 2026
