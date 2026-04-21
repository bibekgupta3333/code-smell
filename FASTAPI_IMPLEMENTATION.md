# FastAPI Frontend Implementation for Code Smell Detection

**Project Status:** Phase 2.5 Complete → Phase 3 (API Layer)  
**Date:** April 20, 2026  
**Goal:** Create REST API for inference pipeline with async support

---

## 📊 Current Completion Status

### ✅ COMPLETED - Core System (Inference Engine)

**Workflow & Orchestration**
- ✅ `src/workflow/workflow_graph.py` - LangGraph state machine (520+ lines)
- ✅ `src/workflow/code_analysis_workflow.py` - Workflow execution
- ✅ `src/workflow/analysis_coordinator.py` - Multi-agent coordination

**LLM & RAG Infrastructure**
- ✅ `src/llm/llm_client.py` - Ollama client wrapper
- ✅ `src/llm/prompt_templates.py` - Prompt management
- ✅ `src/llm/response_parser.py` - Response parsing
- ✅ `src/rag/rag_pipeline.py` - RAG orchestration
- ✅ `src/rag/vector_store.py` - ChromaDB integration
- ✅ `src/rag/embedding_service.py` - Embedding generation

**Code Analysis**
- ✅ `src/analysis/code_parser.py` - Language detection, AST, metrics
- ✅ `src/analysis/code_smell_detector.py` - LangChain ReAct agent
- ✅ `src/analysis/quality_validator.py` - Finding validation
- ✅ `src/analysis/code_chunker.py` - Code chunking

**Supporting Infrastructure**
- ✅ `src/database/database_manager.py` - SQLAlchemy ORM (757 lines, 9 tables)
- ✅ `src/utils/logger.py` - Structured JSON logging
- ✅ `src/utils/result_exporter.py` - Export utilities
- ✅ `src/data/data_loader.py` - Dataset loading
- ✅ `src/data/data_preprocessor.py` - Data normalization

**Baseline Integration**
- ✅ Scripts for running: SonarQube, PMD, Checkstyle, SpotBugs, IntelliJ
- ✅ Result comparison framework

---

## ❌ TO IMPLEMENT - FastAPI REST API Layer

### Phase 3: User-Facing API

#### 3.1 Core API Application
- [ ] Create `src/api_server.py` - Main FastAPI app with dependency injection
- [ ] Create `src/api/models.py` - Pydantic request/response models
- [ ] Create `src/api/routes/__init__.py` - Routes package

#### 3.2 API Endpoints

**3.2.1 Code Analysis Routes** (`src/api/routes/analysis.py`)
- [ ] `POST /api/v1/analyze` - Submit code for analysis
  - Request: code snippet, language (optional), file name (optional)
  - Response: analysis_id, status, metadata
  - Backend: Call `workflow_graph.py`
  
- [ ] `GET /api/v1/results/{analysis_id}` - Retrieve analysis results
  - Response: findings, confidence scores, explanations
  - Backend: Query database_manager
  
- [ ] `GET /api/v1/progress/{analysis_id}` - Real-time progress tracking
  - Response: current_step, percentage, estimated_time_remaining
  - Backend: Query execution state

**3.2.2 Comparison Routes** (`src/api/routes/comparison.py`)
- [ ] `GET /api/v1/compare/{analysis_id}` - Compare LLM vs baseline tools
  - Response: LLM findings, baseline findings, metrics (precision, recall, F1)
  - Backend: Integrate SonarQube/PMD results
  
- [ ] `GET /api/v1/comparison/summary` - Cross-tool summary statistics
  - Response: Aggregated metrics across tools

**3.2.3 Health & Status Routes** (`src/api/routes/health.py`)
- [ ] `GET /api/v1/health` - Service health check
  - Response: status, ollama_available, chromadb_available, database_available
  
- [ ] `GET /api/v1/status` - Detailed system status
  - Response: version, uptime, cache_stats, model_info

#### 3.3 Async Infrastructure
- [ ] `src/api/background_tasks.py` - Background job processing
  - Async analysis queue management
  - Result caching strategy
  - Timeout handling
  
- [ ] `src/api/middleware.py` - Custom middleware
  - Request logging
  - Error handling
  - CORS configuration

#### 3.4 Configuration & Initialization
- [ ] `src/api/config.py` - API-specific settings
  - Server host/port
  - Queue settings
  - Timeout configurations
  
- [ ] `src/api/dependencies.py` - Dependency injection
  - Workflow graph instance
  - Database manager
  - Logger setup

#### 3.5 Error Handling & Validation
- [ ] Create `src/api/exceptions.py` - Custom API exceptions
  - AnalysisTimeout
  - InvalidLanguage
  - InsufficientResources
  
- [ ] Implement `src/api/validators.py` - Request validation
  - Code size limits
  - Language support check
  - Rate limiting

---

## 📋 Request/Response Models

### 3.1 Code Submission

```python
class LanguageEnum(str, Enum):
    JAVA = "java"
    PYTHON = "python"
    JAVASCRIPT = "javascript"

class CodeSubmissionRequest(BaseModel):
    code: str  # Code snippet (1KB - 100KB)
    language: Optional[LanguageEnum] = None  # Auto-detected if None
    file_name: Optional[str] = "code_snippet"
    include_rag: bool = True  # Enable RAG context
    timeout_seconds: int = 300  # 5 minute default

class CodeSubmissionResponse(BaseModel):
    analysis_id: str  # UUID for tracking
    status: str  # "queued" | "processing" | "completed"
    created_at: datetime
    estimated_completion_time: Optional[datetime]
```

### 3.2 Analysis Results

```python
class CodeSmellFindingResponse(BaseModel):
    smell_type: str  # e.g., "Long Method", "God Class"
    location: Dict[str, Any]  # line, column, end_line, end_column
    severity: str  # "low" | "medium" | "high" | "critical"
    confidence: float  # 0.0 - 1.0
    explanation: str  # Why this smell was detected
    rag_context: Optional[List[str]]  # Retrieved examples
    suggested_refactoring: Optional[str]

class AnalysisResultResponse(BaseModel):
    analysis_id: str
    code_hash: str
    language: str
    findings: List[CodeSmellFindingResponse]
    metrics: Dict[str, Any]  # cyclomatic_complexity, lines_of_code, etc.
    analysis_time_ms: float
    model_used: str  # "llama3" | "codelama" | "mistral"
    cache_hit: bool
```

### 3.3 Comparison Results

```python
class BaselineToolResult(BaseModel):
    tool_name: str  # "SonarQube" | "PMD" | "Checkstyle" | "SpotBugs"
    findings_count: int
    true_positives: int
    false_positives: int

class ComparisonResponse(BaseModel):
    analysis_id: str
    llm_results: AnalysisResultResponse
    baseline_results: Dict[str, BaselineToolResult]
    metrics: Dict[str, float]  # precision, recall, f1_score, accuracy
    summary: str  # Key insights
```

---

## 🏗️ API Server Structure

```python
# src/api_server.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import analysis, comparison, health
from src.api.config import settings
from src.api.middleware import setup_middleware

app = FastAPI(
    title="Code Smell Detection API",
    description="Privacy-preserving LLM-based code smell detection",
    version="1.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup custom middleware
setup_middleware(app)

# Include routers
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(comparison.router, prefix="/api/v1", tags=["comparison"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    # Initialize workflow graph
    # Initialize database
    # Warm up embeddings cache

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    # Close database connections
    # Clear temporary cache
```

---

## 🚀 Integration Points

### With Existing Code

**Entry Point:** `src/workflow/workflow_graph.py`
- The API calls the state machine's `ainvoke()` method
- Passes `AnalysisState` with code snippet
- Receives `AnalysisState` with validated findings

**Database:** `src/database/database_manager.py`
- Store analysis metadata
- Store findings with timestamps
- Track experiment runs

**Utilities:** `src/utils/`
- `logger.py` - Structured JSON logging for API calls
- `result_exporter.py` - Export results to CSV/JSON

---

## 📝 Implementation Tasks

### Phase 3.1: Setup & Models (Task 1-3)
- [ ] **Task 1:** Create `src/api/models.py` with all Pydantic models
- [ ] **Task 2:** Create `src/api/config.py` with API configuration
- [ ] **Task 3:** Create `src/api/exceptions.py` with custom exceptions

### Phase 3.2: Routes (Task 4-6)
- [ ] **Task 4:** Create `src/api/routes/analysis.py` with analyze + results endpoints
- [ ] **Task 5:** Create `src/api/routes/comparison.py` with baseline comparison
- [ ] **Task 6:** Create `src/api/routes/health.py` with health checks

### Phase 3.3: Infrastructure (Task 7-8)
- [ ] **Task 7:** Create `src/api/dependencies.py` with DI setup
- [ ] **Task 8:** Create `src/api/middleware.py` with request logging & error handling

### Phase 3.4: Main Application (Task 9)
- [ ] **Task 9:** Create `src/api_server.py` - FastAPI app with all routers

### Phase 3.5: Testing & Documentation (Task 10-11)
- [ ] **Task 10:** Create `src/api/validators.py` with request validation
- [ ] **Task 11:** Generate OpenAPI docs (Swagger UI at `/docs`)

---

## 🔧 Technical Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Framework** | FastAPI | Async native, auto-API docs, Pydantic validation |
| **Async** | asyncio + aiofiles | Non-blocking I/O, better resource usage |
| **Database** | Existing SQLAlchemy ORM | Reuse database_manager.py (9-table schema) |
| **Queuing** | Redis (optional) or in-memory | Handle concurrent requests, avoid analysis duplicates |
| **Caching** | Results cache (session-based) | Fast retrieval of repeat analyses |
| **Timeout** | 300 seconds default | Prevent hanging requests |
| **Rate Limiting** | Token bucket (optional) | Prevent abuse, manage resource usage |

---

## 🔌 Example Usage

```bash
# 1. Start FastAPI server
python -m uvicorn src.api_server:app --reload --port 8000

# 2. Submit code for analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "code": "public class VeryLongMethod { ... }",
    "language": "java",
    "file_name": "VeryLongMethod.java"
  }'

# Response:
# {
#   "analysis_id": "abc-123-def",
#   "status": "processing",
#   "created_at": "2026-04-20T10:30:00Z",
#   "estimated_completion_time": "2026-04-20T10:35:00Z"
# }

# 3. Retrieve results
curl http://localhost:8000/api/v1/results/abc-123-def

# 4. Compare with baselines
curl http://localhost:8000/api/v1/compare/abc-123-def
```

---

## 📚 Dependencies

Ensure these are in `requirements.txt`:

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-multipart==0.0.6
aiofiles==23.2.1
```

---

## 🎯 Success Criteria

- [ ] All 5 API endpoints functional and tested
- [ ] Async inference pipeline handles 10+ concurrent requests
- [ ] Response time < 5 seconds for small snippets (< 1KB)
- [ ] Swagger UI documentation auto-generated at `/docs`
- [ ] Error handling with proper HTTP status codes
- [ ] Results cached and retrievable by analysis_id
- [ ] Integration with existing workflow_graph.py verified

---

## 📌 Next Steps

1. **Immediate:** Implement Task 1-3 (Models, Config, Exceptions)
2. **Short-term:** Implement Task 4-6 (Routes)
3. **Medium-term:** Implement Task 7-9 (Infrastructure & Main App)
4. **Testing:** Create integration tests
5. **Deployment:** Docker containerization + DEPLOYMENT_GUIDE update
