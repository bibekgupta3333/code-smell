# LLM-Based Code Smell Detection System - Dockerfile
# Optimized for reproducibility and M4 Pro architecture compatibility
# Python 3.11 base for consistency across environments
# Build date: February 26, 2026

FROM python:3.11-slim

# Set metadata for version tracking (Architecture Section 11.3)
LABEL maintainer="code-smell-research"
LABEL version="1.0.0"
LABEL description="LLM-based code smell detection system with RAG"
LABEL python.version="3.11"
LABEL build.date="2026-02-26"

# Set environment variables for Python optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
# - git: for cloning repositories if needed
# - build-essential: for compiling Python packages
# - tree-sitter parsers may need build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Note: Ollama runs on host machine (localhost:11434)
# No need to install Ollama in container

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies from pinned requirements
# This ensures reproducibility as per Architecture Section 11.3
RUN pip install --no-cache-dir -r requirements.txt

# Copy project configuration first (for Docker layer caching)
COPY config.py .
COPY .editorconfig .
COPY .cursorrules .

# Create all required directories as per config.py
RUN python config.py

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Copy data directory structure (but not large datasets - mount as volume)
COPY data/.gitkeep ./data/.gitkeep 2>/dev/null || true

# Copy paper and documentation
COPY paper/ ./paper/
COPY docs/ ./docs/
COPY README.md .
COPY PROJECT_STRUCTURE.md .

# Create runtime directories with proper permissions
RUN mkdir -p \
    /app/chromadb_store \
    /app/cache \
    /app/results/predictions \
    /app/results/confusion_matrices \
    /app/results/performance \
    /app/results/resources \
    /app/results/figures \
    /app/results/tables \
    /app/results/metrics \
    /app/results/logs \
    /app/exp/baseline \
    /app/exp/rag_experiments \
    /app/exp/ablation_studies \
    && chmod -R 755 /app

# Expose ports (if using FastAPI or Flask visualization)
# 8000: FastAPI
# 5000: Flask
# 8501: Streamlit
EXPOSE 8000 5000 8501

# Set environment variables for runtime
ENV OLLAMA_BASE_URL=http://ollama:11434 \
    LOG_LEVEL=INFO \
    PYTHONPATH=/app

# Ollama runs on host machine - accessible via host.docker.internal on macOS
ENV OLLAMA_BASE_URL=http://host.docker.internalll (override for specific tasks)
# Example overrides:
#   - Run experiment: docker run <image> python scripts/run_experiment.py
#   - Interactive shell: docker run -it <image> /bin/bash
#   - API server: docker run <image> uvicorn api.main:app --host 0.0.0.0
CMD ["python", "-c", "import config; print('Code Smell Detection System Ready'); print(f'Python: {__import__(\"sys\").version}'); print(f'Ollama URL: {config.OLLAMA_BASE_URL}')"]

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
#
# Build the image:
#   docker build -t code-smell-detector:latest .
#
# Run with Ollama in separate container (recommended):
#   docker network create code-smell-network
#   docker run -d --name ollama --network code-smell-network ollama/ollama:latest
#   docker exec ollama ollama pull llama3:8b
#   docker run --network code-smell-network -v $(pwd)/data:/app/data code-smell-detector
#
# Run standalone (Ollama on host machine):
#   docker run -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#              -v $(pwd)/data:/app/data \
#              -v $(pwd)/results:/app/results \
#              -v $(pwd)/chromadb_store:/app/chromadb_store \
#              code-smell-detector
#
# Interactive development:
#   docker run -it --rm \
#              -v $(pwd)/src:/app/src \
#              -v $(pwd)/scripts:/app/scripts \
#              -v $(pwd)/data:/app/data \
#              -v $(pwd)/results:/app/results \
#              code-smell-detector /bin/bash
#
# Run specific experiment:
#   docker run --network code-smell-network \
#              -v $(pwd)/results:/app/results \
#              code-smell-detector \
#              python scripts/run_baseline.py --experiment-name baseline_001
#
# Docker Compose (recommended for multi-container setup):
#   See docker-compose.yml for full stack configuration
#
# ============================================================================
# VOLUME MOUNT RECOMMENDATIONS
# ============================================================================
#
# Mount these directories as volumes for persistence:
#   - ./data:/app/data                     (datasets - read-only recommended)
#   - ./results:/app/results               (experiment outputs)
#   - ./chromadb_store:/app/chromadb_store (vector database)
#   - ./cache:/app/cache                   (LLM response cache)
#
# ============================================================================
# REPRODUCIBILITY GUARANTEES (Architecture Section 11.3)
# ============================================================================
#
# This Dockerfile ensures reproducibility by:
#   1. Pinning base image to python:3.11-slim (specific Python version)
#   2. Using requirements.txt with pinned dependency versions
#   3. Documenting exact Ollama and model versions in metadata
#   4. Setting fixed random seeds in config.py (RANDOM_SEED=42)
#   5. Controlling environment variables for deterministic behavior
#
# To freeze current dependency versions:
#   pip freeze > requirements-frozen.txt
#
# To verify reproducibility:
#   1. Build image on different machines
#   2. Run same experiment with same input
#   3. Compare output checksums
#
# ============================================================================
