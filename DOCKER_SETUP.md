# Docker Setup Instructions

## Overview

The Docker configuration has been created for the LLM-Based Code Smell Detection System. This includes:

- **Dockerfile**: Reproducible containerized environment (Python dependencies + app)
- **.dockerignore**: Optimized build context
- **docker-compose.yml**: Single-container app orchestration

**Note**: Ollama runs on your host machine (localhost:11434), not in Docker

## Prerequisites

1. **Ollama**: Install and running on your M4 Pro
   - Download from [ollama.ai](https://ollama.ai)
   - Run `ollama serve` or use Ollama Desktop app
   - Models should be accessible at `http://localhost:11434`

2. **Docker Desktop**: Install from [docker.com](https://www.docker.com/products/docker-desktop/)
   - For macOS (M4 Pro): Download "Apple Silicon" version
   - Ensure Docker Desktop is running before building images

3. **Disk Space**: Minimum 10GB free space
   - Base Python image: ~1GB
   - Dependencies: ~2-3GB
   -Prerequisites First

```bash
# 1. Ensure Ollama is running
ollama serve
# Or start Ollama Desktop app

# 2. Verify models are available
ollama list  # Should show llama3:8b and/or codellama:7b

# 3. Test Ollama connectivity
curl http://localhost:11434/api/tags
```

### Docker Compose (Recommended)

```bash
# Start the app container
docker-compose up -d

# Verify app can reach Ollama
docker-compose exec app curl http://host.docker.internal:11434/api/tags

# Run validation script
docker-compose exec app python scripts/validate_config.py

# View logs
# Ensure Ollama is running first
ollama serve  # Run in separate terminal

# Build the image
docker build -t code-smell-detector:latest .

# Run with Ollama on host machine
# On macOS, use host.docker.internal to access host machine
docker run -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
           -v $(pwd)/data:/app/data \
           -v $(pwd)/results:/app/results \
           -v $(pwd)/chromadb_store:/app/chromadb_store \
           code-smell-detector:latest python scripts/validate_config.py

### Option 2: Standalone Docker Build

```bash
# Build the image
docker build -t code-smell-detector:latest .

# Run with Ollama on host machine
docker run -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
           -v $(pwd)/data:/app/data \
           -v $(pwd)/results:/app/results \
           -v $(pwd)/chromadb_store:/app/chromadb_store \
           code-smell-detector:latest

# Run interactive shell
docker run -it --rm code-smell-detector:latest /bin/bash
```

## Development Workflow

### With Live Code Reloading

```bash
# docker-compose.yml already mounts src/ and scripts/ as volumes
docker-compose up -d

# Make changes to code on host machine
# Changes are immediately reflected in container

# Run experiments
docker-compose exec app python scripts/run_experiment.py --experiment-name test
```

### For Production/Reproducibility

```bash
# Comment out development volume mounts in docker-compose.yml
# Rebuild image with code baked in
docker-compose up -d --build

# Code is now frozen in image - fully reproducible
```

## Running Experiments

```bash
# Interactive Python shell
docker-compose exec app python
Since Ollama runs on your host machine, the Docker container uses minimal resources:

- **App Container**: 8GB RAM, 4 CPUs
- **Host Ollama**: Uses remaining system resources

The `docker-compose.yml` is configured for M4 Pro with 16GB RAM. If you have 32GB, increase limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'
      memory: 16
services:
  ollama:
    deploy:
      resources:
        limits:
          cpus: '6.0'
          memory: 12G
  app:
    deploy:
      resources:
        limits:
          cpus: '6.0'
          memory: 12G
```

## Troubleshooting

### Docker Daemon Not Running

```bash
# Error: Cannot connect to the Docker daemon
# SoOllama Not Accessible from Docker

```bash
# Verify Ollama is running on host
ollama list
# or
curl http://localhost:11434/api/tags

# Test connection from inside container
docker-compose exec app curl http://host.docker.internal:11434/api/tags

# If connection fails, start Ollama:
ollama serve  # Run in separate terminal
```bash
# Fix volume permissions
sudo chown -R $(whoami):$(whoami) chromadb_store cache results
```

## Testing Without Docker

If you prefer to run without Docker:

```bash
# Ensure Python 3.11+ and venv are set up
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama separately
# Download from: https://ollama.ai
ollama serve

# Pull models
ollama pull llama3:8b

# Run validation
python scripts/validate_config.py

# Run experiments
python scripts/run_experiment.py
```

## Reproducibility Guarantees

The Docker setup ensures reproducibility by:

1. **Pinned Dependencies**: `requirements.txt` has exact versions
2. **Fixed Base Image**: `python:3.11-slim` 
3. **Documented Versions**: Metadata labels in Dockerfile
4. **Fixed Random Seeds**: `config.py` sets `RANDOM_SEED=42`
5. **Controlled Environment**: Environment variables preset

To verify reproducibility:

```bash
# Build on machine A
docker build -t code-smell-detector:latest .

# Save image
docker save code-smell-detector:latest | gzip > code-smell-detector.tar.gz

# Transfer to machine B, load and run
docker load < code-smell-detector.tar.gz
docker run code-smell-detector:latest python scripts/validate_config.py
```

## Next Steps

After Docker setup is complete:

1. ✅ Configuration Module (Phase 2.1) - **COMPLETE**
2. ⏭️ LLM Integration Module (Phase 2.2) - Next task
3. ⏭️ RAG Implementation (Phase 2.3)
4. ⏭️ Workflow Engine (Phase 2.4)

See `docs/planning/WBS.md` for detailed task breakdown.

## Additional Resources

- **Dockerfile**: Full build instructions with comments
- **docker-compose.yml**: Multi-container orchestration
- **.dockerignore**: Build optimization
- **scripts/validate_config.py**: Validation script for testing setup
- **config.py**: All configuration parameters

For questions or issues, refer to the project documentation in `docs/`.
