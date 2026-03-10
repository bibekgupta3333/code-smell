# src/ Directory Structure

The `src` directory is organized into 7 functional modules for maintainability and clarity.

## Organization

```
src/
├── data/              # Dataset loading & preprocessing
│   ├── data_loader.py         - Load CSV, JSON, XML datasets into unified CodeSample format
│   └── data_preprocessor.py   - Clean, validate, split (60/20/20) code samples
│
├── llm/               # LLM integration & prompting
│   ├── llm_client.py          - Ollama client with caching & retries
│   ├── prompt_templates.py    - System prompts, few-shot examples, RAG templates
│   └── response_parser.py     - Parse LLM JSON responses, validate structure
│
├── rag/               # RAG pipeline components
│   ├── embedding_service.py   - HuggingFace embeddings with disk caching
│   ├── vector_store.py        - ChromaDB operations (add, search, MMR)
│   ├── rag_pipeline.py        - End-to-end RAG workflow
│   └── rag_retriever.py       - Similarity search & context augmentation
│
├── analysis/          # Code analysis & detection
│   ├── code_parser.py         - Language detection, AST parsing, metrics extraction
│   ├── code_chunker.py        - AST-based code chunking with overlap
│   ├── code_smell_detector.py - LangChain Deep Agent for smell detection
│   └── quality_validator.py   - False positive filtering, confidence scoring
│
├── workflow/          # Orchestration & coordination
│   ├── analysis_coordinator.py  - Manager Agent: orchestrates analysis
│   ├── code_analysis_workflow.py - Sequential workflow steps
│   └── workflow_graph.py         - LangGraph state machine for end-to-end analysis
│
├── database/          # Data persistence
│   └── database_manager.py    - SQLAlchemy ORM, experiment tracking
│
└── utils/             # Utilities & helpers
    ├── common.py              - Shared data models & helpers
    ├── logger.py              - Structured logging for experiments
    ├── benchmark_utils.py     - Metrics, stats, quality evaluation
    └── result_exporter.py     - Export to LaTeX, CSV, plots
```

## Module Dependencies

```
┌─────────┐
│  config │  (shared configuration)
└────┬────┘
     │
     ├──► data/         (independent)
     │
     ├──► llm/          (depends on config)
     │    └──► rag/     (depends on llm)
     │
     ├──► analysis/     (depends on llm, rag)
     │
     ├──► workflow/     (depends on analysis, rag, utils)
     │
     ├──► database/     (depends on config)
     │
     └──► utils/        (shared utilities)
```

## Import Examples

**Load datasets:**
```python
from src.data.data_loader import DatasetLoader
loader = DatasetLoader()
samples = loader.load_all()
```

**Preprocess with splits:**
```python
from src.data.data_preprocessor import DataPreprocessor
pp = DataPreprocessor()
split = pp.preprocess_and_split(samples)
```

**Run analysis coordinator:**
```python
from src.workflow.analysis_coordinator import AnalysisCoordinator
coordinator = AnalysisCoordinator()
results = coordinator.coordinate_analysis(code)
```

**Export results:**
```python
from src.utils.result_exporter import to_latex_table, to_csv
to_latex_table(results, output_file='results.tex')
to_csv(results, output_file='results.csv')
```

## Key Classes

| Module | Key Class | Purpose |
|--------|-----------|---------|
| data | `DatasetLoader` | Unified dataset loading |
| data | `DataPreprocessor` | Clean & split datasets |
| llm | `OllamaClient` | LLM API wrapper |
| rag | `EmbeddingService` | Text embeddings |
| rag | `VectorStore` | ChromaDB wrapper |
| rag | `RAGPipeline` | End-to-end RAG |
| analysis | `CodeParser` | Parse code structure |
| analysis | `CodeChunker` | Semantic code chunking |
| analysis | `CodeSmellDetector` | Deep Agent detector |
| workflow | `AnalysisCoordinator` | Manager agent |
| workflow | `WorkflowExecutor` | LangGraph runner |
| database | `DatabaseManager` | Experiment tracking |
| utils | `CodeSampleq / `SmellAnnotation` | Data models |

## Notes

- All imports use absolute paths: `from src.module.submodule import Class`
- Circular dependencies are avoided through careful module ordering
- __init__.py files allow package-level imports
- Old __pycache/ at src root is outdated (use per-subdirectory __pycache__)
