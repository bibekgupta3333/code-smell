# System Architecture
## LLM-Based Code Review System

**Version:** 1.0  
**Last Updated:** February 9, 2026

---

## 1. High-Level Architecture Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Streamlit Web UI]
    end
    
    subgraph "API Gateway Layer"
        API[FastAPI Gateway]
        Auth[Authentication Middleware]
        RateLimit[Rate Limiter]
    end
    
    subgraph "Backend Service Layer"
        Router[API Routes]
        Controller[Controllers]
        Service[Business Logic Services]
    end
    
    subgraph "AI/ML Layer"
        LLM[LLM Service<br/>Ollama Runtime]
        RAG[RAG Pipeline]
        LG[LangGraph Orchestrator]
        Embed[Embedding Service]
    end
    
    subgraph "Data Layer"
        VectorDB[(Vector Store<br/>ChromaDB/FAISS)]
        Cache[(Redis Cache)]
        FileStore[(File Storage)]
    end
    
    subgraph "External Data Sources"
        Dataset[MaRV Dataset]
        Baseline[Baseline Tools<br/>SonarQube/PMD]
    end
    
    UI --> API
    API --> Auth
    Auth --> RateLimit
    RateLimit --> Router
    Router --> Controller
    Controller --> Service
    Service --> LG
    LG --> LLM
    LG --> RAG
    RAG --> Embed
    RAG --> VectorDB
    Service --> Cache
    Service --> FileStore
    Dataset --> VectorDB
    Service --> Baseline
    
    style UI fill:#e1f5ff
    style API fill:#fff4e1
    style LLM fill:#ffe1e1
    style VectorDB fill:#e1ffe1
    style LG fill:#f0e1ff
```

---

## 2. Architecture Layers

### 2.1 Frontend Layer (Streamlit)

**Purpose:** User interface for code submission, review visualization, and results exploration

**Components:**
- **Dashboard Page:** Main entry point, code submission interface
- **Results Viewer:** Display detected code smells with explanations
- **Comparison View:** Side-by-side comparison of LLM vs. baseline tools
- **Analytics Dashboard:** Historical analytics and metrics
- **Settings Panel:** Configuration and preferences

**Technology Stack:**
- Streamlit 1.30+
- Plotly for visualizations
- Pandas for data manipulation
- Requests library for API calls

**Communication:**
- RESTful API calls to FastAPI backend
- HTTP/HTTPS protocol
- JSON data format

---

### 2.2 API Gateway Layer (FastAPI)

**Purpose:** Centralized entry point for all backend services, handles routing, authentication, and rate limiting

**Components:**

1. **API Gateway (FastAPI)**
   - Request routing
   - Response formatting
   - OpenAPI/Swagger documentation
   - CORS configuration

2. **Authentication Middleware**
   - API key validation (optional for MVP)
   - Request authentication
   - User context management

3. **Rate Limiter**
   - Request throttling
   - Quota management
   - DDoS protection

**Endpoints:**
```
POST   /api/v1/code/analyze          # Submit code for analysis
GET    /api/v1/code/results/:id      # Retrieve analysis results
GET    /api/v1/code/history           # Get historical analyses
POST   /api/v1/code/compare           # Compare with baseline tools
GET    /api/v1/smells/types           # List supported smell types
GET    /api/v1/health                 # Health check
```

---

### 2.3 Backend Service Layer

**Purpose:** Business logic, orchestration, and data management

**Structure:**
```
backend/
├── apps/
│   ├── api/              # FastAPI routes and endpoints
│   │   ├── routes/
│   │   │   ├── code_analysis.py
│   │   │   ├── results.py
│   │   │   └── health.py
│   │   └── dependencies.py
│   └── service/          # Business logic
│       ├── code_analyzer.py
│       ├── smell_detector.py
│       ├── result_processor.py
│       └── comparison_service.py
├── core/                 # Core configurations
│   ├── config.py
│   ├── settings.py
│   └── logging_config.py
├── services/             # External integrations
│   ├── llm_service.py
│   ├── rag_service.py
│   ├── embedding_service.py
│   └── vector_store_service.py
├── utils/                # Utilities
│   ├── parsers.py
│   ├── validators.py
│   └── helpers.py
└── models/               # Data models
    ├── requests.py
    ├── responses.py
    └── domain.py
```

**Key Services:**

1. **Code Analyzer Service**
   - Code parsing and preprocessing
   - Language detection
   - AST generation

2. **Smell Detector Service**
   - Orchestrates LLM-based detection
   - Manages detection workflow
   - Aggregates results

3. **Result Processor Service**
   - Formats detection results
   - Calculates metrics
   - Generates explanations

4. **Comparison Service**
   - Runs baseline tools
   - Compares results
   - Statistical analysis

---

### 2.4 AI/ML Layer

**Purpose:** LLM inference, RAG pipeline, and workflow orchestration

**Components:**

#### 2.4.1 LLM Service (Ollama Runtime)

**Models:**
- **Primary:** Llama 3 (8B/13B)
- **Alternative:** CodeLlama (7B/13B)
- **Fallback:** Mistral (7B)

**Capabilities:**
- Code understanding
- Smell detection
- Explanation generation
- Pattern recognition

**Configuration:**
```python
{
    "model": "llama3:8b",
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 2048,
    "context_window": 8192
}
```

#### 2.4.2 RAG Pipeline

**Workflow:**
```mermaid
graph LR
    A[Code Input] --> B[Query Embedding]
    B --> C[Vector Search]
    C --> D[Retrieve Examples]
    D --> E[Context Augmentation]
    E --> F[LLM Prompt]
    F --> G[LLM Inference]
    G --> H[Parsed Output]
```

**Components:**
- **Embedding Service:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store:** ChromaDB or FAISS
- **Retrieval:** Top-K similarity search (K=5)
- **Context Builder:** Formats retrieved examples for prompt

#### 2.4.3 LangGraph Orchestrator

**Detection Workflow:**
```mermaid
graph TD
    Start[Code Input] --> Parse[Parse & Preprocess]
    Parse --> SmellType{Determine Smell Types}
    SmellType --> Retrieve[RAG: Retrieve Examples]
    Retrieve --> Detect[LLM: Detect Smells]
    Detect --> Classify[LLM: Classify Severity]
    Classify --> Reason[LLM: Generate Explanation]
    Reason --> Validate[Validate Output]
    Validate --> Format[Format Results]
    Format --> End[Return Response]
    
    Validate -->|Invalid| Retry[Retry with Refined Prompt]
    Retry --> Detect
```

**State Management:**
```python
class AnalysisState(TypedDict):
    code: str
    language: str
    parsed_ast: dict
    retrieved_examples: List[dict]
    detected_smells: List[dict]
    explanations: List[str]
    confidence_scores: List[float]
    validation_status: str
```

#### 2.4.4 Embedding Service

**Model:** sentence-transformers/all-MiniLM-L6-v2
- Dimension: 384
- Fast inference
- Good semantic understanding
- Free and open-source

**Usage:**
- Embed code snippets for vector search
- Embed smell descriptions
- Similarity calculation

---

### 2.5 Data Layer

#### 2.5.1 Vector Store (ChromaDB/FAISS)

**Purpose:** Store and retrieve code smell examples

**Collections:**
```
smell_examples/
├── long_method_examples
├── large_class_examples
├── feature_envy_examples
├── data_clumps_examples
└── ...
```

**Document Structure:**
```json
{
  "id": "smell_001",
  "smell_type": "Long Method",
  "code_snippet": "...",
  "explanation": "...",
  "severity": "high",
  "metadata": {
    "language": "java",
    "source": "marv_dataset",
    "validated": true
  },
  "embedding": [0.123, 0.456, ...]
}
```

#### 2.5.2 Cache Layer (Redis - Optional for MVP)

**Purpose:** Cache LLM responses for identical code inputs

**Cache Strategy:**
- **Key:** SHA256 hash of code + model + parameters
- **Value:** Detection results JSON
- **TTL:** 7 days

**Benefits:**
- Reduce redundant LLM calls
- Faster response times
- Cost savings

#### 2.5.3 File Storage

**Purpose:** Store uploaded code files and analysis results

**Structure:**
```
storage/
├── uploads/          # Temporary uploaded files
├── results/          # Analysis results (JSON)
└── datasets/         # MaRV dataset cache
```

---

## 3. Data Flow Diagrams

### 3.1 Code Analysis Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant API as FastAPI
    participant Service as Analysis Service
    participant LG as LangGraph
    participant RAG as RAG Pipeline
    participant LLM as Ollama LLM
    participant VDB as Vector Store
    
    User->>UI: Submit Code
    UI->>API: POST /api/v1/code/analyze
    API->>Service: analyze_code(code)
    Service->>LG: start_analysis_workflow(code)
    LG->>LG: parse_and_preprocess()
    LG->>RAG: retrieve_examples(code)
    RAG->>VDB: similarity_search(embedding)
    VDB-->>RAG: top_k_examples
    RAG-->>LG: augmented_context
    LG->>LLM: detect_smells(code + context)
    LLM-->>LG: detected_smells
    LG->>LLM: generate_explanations(smells)
    LLM-->>LG: explanations
    LG->>LG: validate_and_format()
    LG-->>Service: analysis_results
    Service-->>API: formatted_response
    API-->>UI: JSON response
    UI-->>User: Display Results
```

### 3.2 Comparison Flow

```mermaid
sequenceDiagram
    participant User
    participant UI
    participant API
    participant Service
    participant LLM as LLM Analysis
    participant Baseline as Baseline Tools
    
    User->>UI: Request Comparison
    UI->>API: POST /api/v1/code/compare
    
    par LLM Analysis
        API->>Service: run_llm_analysis()
        Service->>LLM: detect_smells()
        LLM-->>Service: llm_results
    and Baseline Analysis
        API->>Service: run_baseline_tools()
        Service->>Baseline: analyze_code()
        Baseline-->>Service: baseline_results
    end
    
    Service->>Service: compare_results()
    Service->>Service: calculate_metrics()
    Service-->>API: comparison_data
    API-->>UI: JSON response
    UI-->>User: Display Comparison
```

---

## 4. Technology Stack Summary

### Frontend
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| UI Framework | Streamlit | 1.30+ | Web interface |
| Visualization | Plotly | 5.18+ | Charts and graphs |
| Data | Pandas | 2.1+ | Data manipulation |
| HTTP Client | Requests | 2.31+ | API communication |

### Backend
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Web Framework | FastAPI | 0.109+ | REST API |
| ASGI Server | Uvicorn | 0.27+ | Production server |
| Validation | Pydantic | 2.5+ | Data validation |
| Async | AsyncIO | Built-in | Asynchronous operations |

### AI/ML
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| LLM Runtime | Ollama | Latest | Local LLM hosting |
| Models | Llama 3, CodeLlama | Latest | Code understanding |
| Orchestration | LangGraph | 0.0.20+ | Workflow management |
| Embeddings | sentence-transformers | 2.3+ | Vector embeddings |
| Vector Store | ChromaDB | 0.4.22+ | Similarity search |
| Alternative VDB | FAISS | 1.7+ | Vector search (optional) |

### Data
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Vector DB | ChromaDB | 0.4.22+ | Embedding storage |
| Cache | Redis | 7.2+ | Response caching (optional) |
| File Storage | Local FS | - | File persistence |

### DevOps
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Containerization | Docker | 24.0+ | Application packaging |
| Orchestration | Docker Compose | 2.24+ | Multi-container setup |
| Version Control | Git | 2.40+ | Source control |
| CI/CD | GitHub Actions | - | Automation |

---

## 5. Deployment Architecture

```mermaid
graph TB
    subgraph "Docker Host"
        subgraph "Frontend Container"
            ST[Streamlit App<br/>Port 8501]
        end
        
        subgraph "Backend Container"
            FA[FastAPI Server<br/>Port 8000]
        end
        
        subgraph "LLM Container"
            OL[Ollama Service<br/>Port 11434]
        end
        
        subgraph "Vector Store Container"
            CH[ChromaDB<br/>Port 8001]
        end
        
        subgraph "Cache Container (Optional)"
            RD[Redis<br/>Port 6379]
        end
        
        subgraph "Volumes"
            V1[(Models Volume)]
            V2[(Vector Data)]
            V3[(Upload Storage)]
        end
    end
    
    ST -->|HTTP| FA
    FA -->|HTTP| OL
    FA -->|HTTP| CH
    FA -.->|Optional| RD
    OL --> V1
    CH --> V2
    FA --> V3
    
    User[User Browser] -->|HTTPS| ST
    
    style ST fill:#e1f5ff
    style FA fill:#fff4e1
    style OL fill:#ffe1e1
    style CH fill:#e1ffe1
```

**Container Specifications:**

| Container | Base Image | CPU | Memory | Storage |
|-----------|-----------|-----|--------|---------|
| Frontend | python:3.11-slim | 0.5 | 512MB | - |
| Backend | python:3.11-slim | 1.0 | 1GB | 1GB |
| Ollama | ollama/ollama:latest | 2.0 | 8GB | 20GB |
| ChromaDB | chromadb/chroma:latest | 1.0 | 2GB | 10GB |
| Redis | redis:7-alpine | 0.5 | 256MB | 1GB |

---

## 6. Security Architecture

### 6.1 Security Layers

```mermaid
graph TB
    Internet[Internet]
    
    subgraph "Security Perimeter"
        FW[Firewall]
        HTTPS[HTTPS/TLS]
        
        subgraph "Application Security"
            Auth[Authentication]
            RateLimit[Rate Limiting]
            InputVal[Input Validation]
            Sanitize[Code Sanitization]
        end
        
        subgraph "Data Security"
            Encrypt[Data Encryption]
            AccessCtrl[Access Control]
        end
    end
    
    Internet --> FW
    FW --> HTTPS
    HTTPS --> Auth
    Auth --> RateLimit
    RateLimit --> InputVal
    InputVal --> Sanitize
    Sanitize --> Encrypt
    Encrypt --> AccessCtrl
```

### 6.2 Security Measures

**Network Security:**
- HTTPS/TLS for all external communication
- Internal Docker network isolation
- Firewall rules for container communication

**Application Security:**
- Input validation for all API requests
- Code sanitization before LLM processing
- Rate limiting per IP/API key
- CORS configuration
- XSS protection
- SQL injection prevention (if using SQL database)

**Data Security:**
- No storage of sensitive code without explicit permission
- Temporary file cleanup after analysis
- Optional encryption at rest for stored results
- Access control for historical data

**LLM Security:**
- Prompt injection prevention
- Output validation and sanitization
- Model isolation in container
- Resource limits to prevent DoS

---

## 7. Scalability Considerations

### 7.1 Horizontal Scaling

**Scalable Components:**
- **Frontend:** Multiple Streamlit instances behind load balancer
- **Backend:** Multiple FastAPI workers via Uvicorn
- **Vector Store:** ChromaDB supports distributed mode

**Non-Scalable (Current Architecture):**
- **Ollama:** Single instance per GPU (future: model serving frameworks)

### 7.2 Vertical Scaling

**Resource Optimization:**
- Model quantization (4-bit, 8-bit) for lower memory
- Batch processing for multiple files
- Connection pooling for database connections
- Caching frequently accessed data

### 7.3 Performance Optimization

**Backend:**
- Async FastAPI endpoints
- Connection pooling
- Response caching (Redis)
- Database query optimization

**LLM Inference:**
- Model quantization
- Batch inference where possible
- Streaming responses for UI feedback
- Model warm-up on startup

**Vector Search:**
- Index optimization
- Approximate nearest neighbor search
- Pre-compute embeddings for static data

---

## 8. Monitoring and Observability

### 8.1 Logging

**Log Levels:**
- **ERROR:** System errors, LLM failures
- **WARNING:** Rate limit hits, validation failures
- **INFO:** Request/response, analysis started/completed
- **DEBUG:** Detailed execution flow

**Log Destinations:**
- Container stdout (Docker logs)
- File logs (persistent volume)
- Optional: Centralized logging (future)

### 8.2 Metrics

**Application Metrics:**
- Request count, response times
- Error rates, success rates
- Active users, concurrent requests

**LLM Metrics:**
- Inference time per request
- Tokens processed
- Model cache hit rate
- Hallucination detection rate

**System Metrics:**
- CPU, memory, disk usage
- Container health status
- Network throughput

### 8.3 Health Checks

**Endpoints:**
```
GET /api/v1/health              # Basic health
GET /api/v1/health/ready        # Readiness probe
GET /api/v1/health/live         # Liveness probe
```

**Health Check Components:**
- API server status
- Ollama connectivity
- Vector store connectivity
- Disk space availability

---

## 9. Error Handling and Resilience

### 9.1 Error Handling Strategy

```mermaid
graph TD
    Request[Incoming Request] --> Validate{Valid Input?}
    Validate -->|No| ReturnError[Return 400 Error]
    Validate -->|Yes| Process[Process Request]
    Process --> LLMCall{LLM Available?}
    LLMCall -->|No| Retry{Retry Count < 3?}
    Retry -->|Yes| Wait[Wait & Retry]
    Wait --> LLMCall
    Retry -->|No| Fallback[Use Fallback Response]
    LLMCall -->|Yes| Inference[Run Inference]
    Inference --> ValidateResp{Valid Response?}
    ValidateResp -->|No| Retry
    ValidateResp -->|Yes| Success[Return Success]
    Fallback --> PartialSuccess[Return Partial Results]
```

### 9.2 Resilience Patterns

**Retry Logic:**
- Exponential backoff for LLM calls
- Maximum 3 retries
- Different strategies per error type

**Circuit Breaker:**
- Protect against cascading failures
- Open circuit after 5 consecutive failures
- Half-open state for recovery testing

**Graceful Degradation:**
- Return partial results if possible
- Use cached results when LLM unavailable
- Inform user of degraded service

**Timeout Management:**
- Request timeout: 60 seconds
- LLM inference timeout: 45 seconds
- Database query timeout: 5 seconds

---

## 10. Future Architecture Enhancements

### Short-Term (Next 3 Months)
- [ ] Redis caching implementation
- [ ] Metrics dashboard (Prometheus + Grafana)
- [ ] API authentication (JWT)
- [ ] Enhanced logging (structured logs)

### Medium-Term (3-6 Months)
- [ ] Microservices architecture (separate LLM service)
- [ ] Message queue (RabbitMQ/Kafka) for async processing
- [ ] Multi-model support (ensemble)
- [ ] WebSocket for real-time updates

### Long-Term (6-12 Months)
- [ ] Kubernetes deployment
- [ ] Distributed vector store
- [ ] Model fine-tuning pipeline
- [ ] Multi-tenancy support
- [ ] Advanced analytics and ML ops

---

## 11. Architecture Decision Records (ADRs)

### ADR-001: Use Ollama for Local LLM Hosting
**Status:** Accepted  
**Decision:** Use Ollama instead of custom LLM serving  
**Rationale:** Easy setup, model management, wide model support  
**Consequences:** Limited to Ollama-compatible models

### ADR-002: FastAPI for Backend
**Status:** Accepted  
**Decision:** Use FastAPI instead of Flask/Django  
**Rationale:** Modern, async, auto-documentation, type hints  
**Consequences:** Python 3.7+ required

### ADR-003: ChromaDB for Vector Store
**Status:** Accepted  
**Decision:** Use ChromaDB over FAISS  
**Rationale:** Easier management, built-in persistence, better DX  
**Consequences:** Slightly slower than optimized FAISS

### ADR-004: Monorepo Structure
**Status:** Accepted  
**Decision:** Monorepo with backend + frontend  
**Rationale:** Easier development, shared types, atomic commits  
**Consequences:** Larger repository size

### ADR-005: LangGraph for Orchestration
**Status:** Accepted  
**Decision:** Use LangGraph for workflow management  
**Rationale:** Structured workflows, state management, debuggability  
**Consequences:** Additional dependency, learning curve

---

## 12. Architecture Diagram Legend

**Color Coding:**
- 🔵 Blue: User-facing components
- 🟡 Yellow: API/Gateway layer
- 🔴 Red: AI/ML components
- 🟢 Green: Data storage
- 🟣 Purple: Orchestration/Workflow

**Arrows:**
- Solid: Synchronous communication
- Dashed: Asynchronous/Optional communication
- Thick: High-traffic paths

---

**Document Version:** 1.0  
**Last Review:** February 9, 2026  
**Next Review:** March 1, 2026  
**Maintained By:** Architecture Team
