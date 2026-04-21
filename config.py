"""
Configuration for LLM-Based Code Smell Detection System
Optimized for M4 Pro laptop (16GB+ RAM, CPU inference)
"""

import os
from pathlib import Path
from typing import Literal

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MARV_DATASET_DIR = DATASETS_DIR / "marv"
QUALITAS_CORPUS_DIR = DATASETS_DIR / "qualitas_corpus"
SMELLY_CODE_DIR = DATASETS_DIR / "smelly_code"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
PERFORMANCE_DIR = RESULTS_DIR / "performance"
RESOURCES_DIR = RESULTS_DIR / "resources"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"

# Experiment directories
EXP_DIR = PROJECT_ROOT / "exp"
BASELINE_EXP_DIR = EXP_DIR / "baseline"
RAG_EXP_DIR = EXP_DIR / "rag_experiments"
ABLATION_DIR = EXP_DIR / "ablation_studies"

# Other directories
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
VISUALIZATION_DIR = PROJECT_ROOT / "visualization"
PAPER_DIR = PROJECT_ROOT / "paper"

# ============================================================================
# OLLAMA CONFIGURATION (Local LLM)
# ============================================================================

# Ollama API configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds

# Default model selection (optimized for M4 Pro - 8B models for efficiency)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3:8b")

# Available models for different use cases
AVAILABLE_MODELS = {
    "fast": "llama3:8b",           # Fast, efficient - primary choice for M4 Pro
    "accurate": "llama3:13b",      # Better accuracy, slower - use sparingly
    "code_specialized": "codellama:7b",  # Code-focused, long context
    "code_advanced": "codellama:13b",    # Best code understanding - resource intensive
}

# Model selection strategy parameters
MODEL_SELECTION = {
    "max_tokens_fast": 2048,        # Use fast model if input < this
    "max_tokens_accurate": 4096,    # Switch to accurate model
    "complexity_threshold": 10,     # Cyclomatic complexity threshold
}

# LLM inference parameters (conservative for reproducibility)
LLM_CONFIG = {
    "temperature": 0.1,             # Low for deterministic outputs
    "top_p": 0.9,
    "max_tokens": 2048,
    "seed": 42,                     # Fixed for reproducibility
    "num_ctx": 4096,               # Context window
    "repeat_penalty": 1.1,
}

# ============================================================================
# RAG CONFIGURATION (ChromaDB + Embeddings)
# ============================================================================

# ChromaDB persistent storage (optimized for M4 Pro)
CHROMADB_DIR = PROJECT_ROOT / "chromadb_store"
CHROMADB_COLLECTION_NAME = "code_smell_examples"

# Embedding model configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight, 384-dim
)
EMBEDDING_DIMENSION = 384
EMBEDDING_DEVICE = "cpu"  # M4 Pro - use CPU (efficient integrated GPU handling)

# RAG retrieval parameters (from Architecture Section 3)
RAG_CONFIG = {
    "top_k": 5,                     # Default retrieval count (Architecture default)
    "similarity_threshold": 0.7,    # Minimum similarity score
    "diversity_lambda": 0.7,        # MMR diversity parameter
    "max_context_length": 2048,     # Max tokens for retrieved context
    "enable_reranking": True,       # Enable MMR reranking
}

# ============================================================================
# DATABASE CONFIGURATION (SQLite for agent tracking)
# ============================================================================

DATABASE_PATH = PROJECT_ROOT / "code_smell_experiments.db"
DATABASE_CONFIG = {
    "check_same_thread": False,     # Allow multi-threaded access
    "timeout": 30,                  # Lock timeout in seconds
}

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Dataset split ratios (Benchmarking Section 1)
DATA_SPLIT = {
    "train": 0.60,      # For RAG knowledge base
    "validation": 0.20, # For hyperparameter tuning
    "test": 0.20,       # For final evaluation
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Batch processing (optimized for M4 Pro memory)
BATCH_SIZE = 5  # Process 5 code samples at a time
MAX_CONCURRENT_REQUESTS = 2  # Limit concurrent LLM calls

# ============================================================================
# CODE ANALYSIS CONFIGURATION
# ============================================================================

# Code chunking parameters
CODE_CHUNKING = {
    "max_chunk_tokens": 512,        # AST-based chunking limit
    "overlap_tokens": 50,           # Overlap between chunks
    "preserve_structure": True,     # Respect function boundaries
}

# Supported programming languages
SUPPORTED_LANGUAGES = ["java", "python", "javascript", "cpp"]

# Code smell types (Production code smells - SonarQube + Fowler catalog)
# NOTE: The authoritative catalog lives in src/utils/smell_catalog.py.
# This list is re-exported here for backward compatibility with downstream
# modules (data loader, evaluator, result exporter).
CODE_SMELL_TYPES = [
    # Bloaters
    "Long Method",
    "God Class",
    "Large Class",
    "Long Parameter List",
    "Primitive Obsession",
    "Data Clumps",
    # OO Abusers
    "Switch Statements",
    "Refused Bequest",
    "Temporary Field",
    "Alternative Classes with Different Interfaces",
    # Change Preventers
    "Divergent Change",
    "Shotgun Surgery",
    "Parallel Inheritance Hierarchies",
    # Dispensables
    "Duplicate Code",
    "Lazy Class",
    "Data Class",
    "Dead Code",
    "Speculative Generality",
    "Comments",
    # Couplers
    "Feature Envy",
    "Inappropriate Intimacy",
    "Message Chains",
    "Middle Man",
    # Complexity & Quality (SonarQube-style)
    "High Cyclomatic Complexity",
    "Deep Nesting",
    "Magic Numbers",
    "Inconsistent Naming",
    "Missing Error Handling",
    "Empty Catch Block",
]

# ============================================================================
# PERFORMANCE & BENCHMARKING CONFIGURATION
# ============================================================================

# Performance targets (Architecture Section 1.1)
PERFORMANCE_TARGETS = {
    "latency_max_seconds": 5,           # Max end-to-end latency
    "throughput_min_per_hour": 100,     # Min analyses per hour
    "memory_max_gb": 8,                 # Max memory usage (M4 Pro safe limit)
    "hallucination_rate_max": 0.05,     # Max 5% hallucination rate
    "cache_hit_rate_min": 0.20,         # Min 20% cache hit rate
}

# Resource monitoring interval
RESOURCE_MONITORING_INTERVAL = 1.0  # seconds

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "code_smell_detector.log"

# Structured logging for experiments
ENABLE_STRUCTURED_LOGGING = True
EXPERIMENT_LOG_DIR = RESULTS_DIR / "logs"

# ============================================================================
# BASELINE TOOLS CONFIGURATION
# ============================================================================

BASELINE_TOOLS = {
    "sonarqube": {
        "enabled": True,
        "timeout": 60,
        "version": "10.x",
    },
    "pmd": {
        "enabled": True,
        "timeout": 60,
        "version": "7.x",
    },
    "checkstyle": {
        "enabled": True,
        "timeout": 60,
        "version": "10.x",
    },
    "spotbugs": {
        "enabled": True,
        "timeout": 60,
        "version": "4.x",
    },
    "intellij": {
        "enabled": False,  # Requires manual setup
        "timeout": 60,
        "version": "2024.x",
    },
}

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

# LLM response caching (Architecture Section 9.1)
ENABLE_LLM_CACHE = True
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_MAX_SIZE_MB = 500  # Max cache size in MB
CACHE_TTL_HOURS = 24     # Cache time-to-live

# ============================================================================
# RETRY & ERROR HANDLING
# ============================================================================

# Retry configuration for LLM calls
RETRY_CONFIG = {
    "max_retries": 3,
    "initial_delay": 1.0,       # seconds
    "max_delay": 10.0,          # seconds
    "exponential_base": 2,
}

# Fallback model chain
FALLBACK_MODELS = [
    "llama3:8b",       # Primary
    "codellama:7b",    # Fallback 1
]

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

# Response validation thresholds
VALIDATION_CONFIG = {
    "min_confidence": 0.6,              # Minimum confidence to accept detection
    "max_hallucination_check": True,    # Enable hallucination detection
    "require_valid_json": True,         # Require valid JSON response
    "max_retries_on_invalid": 2,        # Retries for invalid responses
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR, DATASETS_DIR, MARV_DATASET_DIR, QUALITAS_CORPUS_DIR,
        SMELLY_CODE_DIR, GROUND_TRUTH_DIR, RESULTS_DIR, PREDICTIONS_DIR,
        CONFUSION_MATRICES_DIR, PERFORMANCE_DIR, RESOURCES_DIR, FIGURES_DIR,
        TABLES_DIR, METRICS_DIR, EXP_DIR, BASELINE_EXP_DIR, RAG_EXP_DIR,
        ABLATION_DIR, SCRIPTS_DIR, VISUALIZATION_DIR, PAPER_DIR,
        CHROMADB_DIR, CACHE_DIR, EXPERIMENT_LOG_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_model_for_task(
    code_length: int,
    complexity: int = 5,
    priority: Literal["fast", "accurate"] = "fast"
) -> str:
    """
    Select appropriate model based on task requirements.

    Args:
        code_length: Length of code in tokens
        complexity: Code complexity score
        priority: Priority level (fast for experiments, accurate for finals)

    Returns:
        Model name to use
    """
    if priority == "accurate" or complexity > MODEL_SELECTION["complexity_threshold"]:
        if code_length > MODEL_SELECTION["max_tokens_accurate"]:
            return AVAILABLE_MODELS["code_advanced"]
        return AVAILABLE_MODELS["accurate"]

    if code_length > MODEL_SELECTION["max_tokens_fast"]:
        return AVAILABLE_MODELS["code_specialized"]

    return AVAILABLE_MODELS["fast"]


if __name__ == "__main__":
    # Create all directories when config is run directly
    ensure_directories()
    print("✓ All project directories created successfully")
    print(f"✓ Project root: {PROJECT_ROOT}")
    print(f"✓ Ollama URL: {OLLAMA_BASE_URL}")
    print(f"✓ Default model: {DEFAULT_MODEL}")
    print(f"✓ Embedding model: {EMBEDDING_MODEL}")
    print(f"✓ Database: {DATABASE_PATH}")
