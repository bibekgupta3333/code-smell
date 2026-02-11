# Deployment Guide
## LLM-Based Code Review System - Step-by-Step Deployment

**Version:** 1.0  
**Last Updated:** February 9, 2026

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Local Development Setup](#2-local-development-setup)
3. [Docker Deployment](#3-docker-deployment)
4. [Configuration](#4-configuration)
5. [Data Initialization](#5-data-initialization)
6. [Running the Application](#6-running-the-application)
7. [Verification & Testing](#7-verification--testing)
8. [Troubleshooting](#8-troubleshooting)
9. [Production Deployment](#9-production-deployment)

---

## 1. Prerequisites

### 1.1 Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 16 GB
- Storage: 50 GB free space
- GPU: Not required (CPU-only operation supported)

**Recommended:**
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 100 GB SSD
- GPU: NVIDIA GPU with 8+ GB VRAM (for faster LLM inference)

### 1.2 Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| **Docker** | 24.0+ | Containerization |
| **Docker Compose** | 2.24+ | Multi-container orchestration |
| **Git** | 2.40+ | Version control |
| **Python** | 3.11+ | Local development (optional) |
| **Make** | 4.0+ | Build automation (optional) |

### 1.3 Operating System Support

- ✅ **Linux** (Ubuntu 22.04+, Debian 11+)
- ✅ **macOS** (Monterey 12+)
- ✅ **Windows** (10/11 with WSL2)

---

## 2. Local Development Setup

### 2.1 Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/code-review-llm.git
cd code-review-llm

# Verify structure
ls -la
```

**Expected Structure:**
```
code-review-llm/
├── backend/
├── frontend/
├── docker/
├── docs/
├── docker-compose.yml
├── Makefile
└── README.md
```

### 2.2 Install Docker

**Ubuntu/Debian:**
```bash
# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

**macOS:**
```bash
# Install Docker Desktop
brew install --cask docker

# Or download from: https://www.docker.com/products/docker-desktop/

# Start Docker Desktop application

# Verify
docker --version
docker compose version
```

**Windows (WSL2):**
```powershell
# Install WSL2 first
wsl --install

# Install Docker Desktop for Windows
# Download from: https://www.docker.com/products/docker-desktop/

# Enable WSL2 backend in Docker Desktop settings

# Verify (in WSL2 terminal)
docker --version
docker compose version
```

### 2.3 Configure Docker (Optional)

**Add user to docker group (Linux):**
```bash
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

**Configure Docker resources:**
- Docker Desktop → Settings → Resources
  - CPUs: 4-8 cores
  - Memory: 8-16 GB
  - Swap: 2 GB
  - Disk: 60 GB

---

## 3. Docker Deployment

### 3.1 Project Structure

```
code-review-llm/
├── backend/
│   ├── apps/
│   ├── core/
│   ├── services/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker/
│   ├── backend/
│   │   └── Dockerfile
│   ├── frontend/
│   │   └── Dockerfile
│   └── nginx/
│       └── nginx.conf
├── docker-compose.yml
└── .env.example
```

### 3.2 Docker Compose Configuration

**docker-compose.yml:**
```yaml
version: '3.9'

services:
  # Backend API Service
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: codereview-backend
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8001
      - REDIS_HOST=redis
      - LOG_LEVEL=INFO
    env_file:
      - .env
    volumes:
      - ./backend:/app
      - backend-storage:/app/storage
    depends_on:
      - ollama
      - chromadb
      - redis
    networks:
      - codereview-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Frontend Streamlit Service
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: codereview-frontend
    ports:
      - "8501:8501"
    environment:
      - BACKEND_API_URL=http://backend:8000
    env_file:
      - .env
    volumes:
      - ./frontend:/app
    depends_on:
      - backend
    networks:
      - codereview-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Ollama LLM Service
  ollama:
    image: ollama/ollama:latest
    container_name: codereview-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama
    networks:
      - codereview-network
    restart: unless-stopped
    # Uncomment for GPU support (NVIDIA)
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

  # ChromaDB Vector Store
  chromadb:
    image: chromadb/chroma:latest
    container_name: codereview-chromadb
    ports:
      - "8001:8000"
    volumes:
      - chromadb-data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
    networks:
      - codereview-network
    restart: unless-stopped

  # Redis Cache (Optional)
  redis:
    image: redis:7-alpine
    container_name: codereview-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    networks:
      - codereview-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  ollama-models:
    driver: local
  chromadb-data:
    driver: local
  backend-storage:
    driver: local
  redis-data:
    driver: local

networks:
  codereview-network:
    driver: bridge
```

### 3.3 Environment Configuration

**Create `.env` file:**
```bash
cp .env.example .env
```

**Edit `.env`:**
```bash
# Application
APP_NAME=Code Review LLM System
DEBUG=false
ALLOWED_ORIGINS=http://localhost:8501,http://localhost:3000

# LLM Configuration
OLLAMA_BASE_URL=http://ollama:11434
DEFAULT_LLM_MODEL=llama3:8b
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2048

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Vector Store
CHROMA_HOST=chromadb
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=code_smell_examples

# RAG Configuration
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.7

# Cache
REDIS_HOST=redis
REDIS_PORT=6379
CACHE_TTL=604800

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=60
```

### 3.4 Dockerfiles

**backend/Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create storage directories
RUN mkdir -p /app/storage/uploads /app/storage/results /app/storage/chromadb

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**frontend/Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 4. Build and Start Services

### 4.1 Build Docker Images

```bash
# Build all services
docker compose build

# Or build specific service
docker compose build backend
docker compose build frontend
```

**Expected Output:**
```
[+] Building 45.2s (12/12) FINISHED
 => [backend internal] load build definition
 => [backend] transferring dockerfile
 ...
 => [frontend] exporting to image
Successfully built and tagged codereview-backend:latest
Successfully built and tagged codereview-frontend:latest
```

### 4.2 Start Services

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f backend
```

**Expected Output:**
```
[+] Running 5/5
 ✔ Container codereview-redis     Started
 ✔ Container codereview-chromadb  Started
 ✔ Container codereview-ollama    Started
 ✔ Container codereview-backend   Started
 ✔ Container codereview-frontend  Started
```

### 4.3 Verify Services

```bash
# Check running containers
docker compose ps

# Expected output:
# NAME                     STATUS              PORTS
# codereview-backend       Up (healthy)        0.0.0.0:8000->8000/tcp
# codereview-frontend      Up (healthy)        0.0.0.0:8501->8501/tcp
# codereview-ollama        Up                  0.0.0.0:11434->11434/tcp
# codereview-chromadb      Up                  0.0.0.0:8001->8000/tcp
# codereview-redis         Up (healthy)        0.0.0.0:6379->6379/tcp
```

---

## 5. Data Initialization

### 5.1 Download LLM Models

```bash
# Enter Ollama container
docker exec -it codereview-ollama bash

# Pull models
ollama pull llama3:8b
ollama pull codellama:7b
ollama pull mistral:7b

# Verify models
ollama list

# Exit container
exit
```

**Expected Output:**
```
NAME                    ID              SIZE    MODIFIED
llama3:8b               a6990ed6be41    4.7 GB  2 minutes ago
codellama:7b            8fdf8f752f6e    3.8 GB  5 minutes ago
mistral:7b              f974a74358d6    4.1 GB  7 minutes ago
```

### 5.2 Initialize Vector Store

```bash
# Run initialization script
docker exec -it codereview-backend python scripts/init_database.py
```

**Expected Output:**
```
[INFO] Initializing database...
[INFO] Creating ChromaDB collections...
[INFO] Collection 'code_smell_examples' created
[INFO] Collection 'refactoring_patterns' created
[INFO] Indexing MaRV dataset...
[INFO] Indexed 150 examples for Long Method
[INFO] Indexed 120 examples for Large Class
[INFO] Database initialization complete!
```

### 5.3 Load MaRV Dataset (Optional)

```bash
# Download MaRV dataset
cd storage/datasets
git clone https://github.com/HRI-EU/SmellyCodeDataset.git marv

# Index dataset
docker exec -it codereview-backend python scripts/index_marv.py
```

---

## 6. Running the Application

### 6.1 Access Services

**Frontend (Streamlit):**
- URL: http://localhost:8501
- Description: Web interface for code analysis

**Backend API:**
- URL: http://localhost:8000
- Docs: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

**Ollama API:**
- URL: http://localhost:11434

**ChromaDB:**
- URL: http://localhost:8001

### 6.2 Test API Endpoint

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Expected response:
# {"status":"ok","timestamp":"2026-02-09T10:30:00Z"}

# Analyze code
curl -X POST http://localhost:8000/api/v1/code/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "code": "public class Example { public void longMethod() { /* 100 lines */ } }",
    "language": "java",
    "analysis_mode": "thorough"
  }'
```

### 6.3 Test Frontend

1. Open browser: http://localhost:8501
2. Paste code snippet
3. Click "Analyze Code"
4. View results

---

## 7. Verification & Testing

### 7.1 Service Health Checks

```bash
# Check all services
docker compose ps

# Check backend health
curl http://localhost:8000/api/v1/health

# Check frontend health
curl http://localhost:8501/_stcore/health

# Check Ollama
curl http://localhost:11434/api/tags

# Check ChromaDB
curl http://localhost:8001/api/v1/heartbeat

# Check Redis
docker exec -it codereview-redis redis-cli ping
```

### 7.2 End-to-End Test

```bash
# Run test suite
docker exec -it codereview-backend pytest tests/integration/

# Expected output:
# ======================== test session starts =========================
# collected 15 items
# tests/integration/test_api.py .................. [ 100%]
# ======================== 15 passed in 45.2s ==========================
```

### 7.3 Performance Test

```bash
# Install testing tools
pip install locust

# Run load test
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

---

## 8. Troubleshooting

### 8.1 Common Issues

**Issue: Container fails to start**
```bash
# Check logs
docker compose logs backend

# Common causes:
# - Port already in use
# - Missing environment variables
# - Insufficient resources

# Solution:
docker compose down
docker compose up -d
```

**Issue: Ollama model not found**
```bash
# Enter container and pull model
docker exec -it codereview-ollama ollama pull llama3:8b
```

**Issue: ChromaDB connection error**
```bash
# Restart ChromaDB
docker compose restart chromadb

# Check logs
docker compose logs chromadb
```

**Issue: Out of memory**
```bash
# Increase Docker memory limit (Docker Desktop)
# Settings → Resources → Memory → 16 GB

# Or use smaller models
# llama3:8b instead of llama3:70b
```

### 8.2 Logs

```bash
# View all logs
docker compose logs

# Follow logs
docker compose logs -f

# View specific service
docker compose logs -f backend

# View last 100 lines
docker compose logs --tail=100 backend
```

### 8.3 Restart Services

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart backend

# Stop and start (full restart)
docker compose down
docker compose up -d
```

---

## 9. Production Deployment

### 9.1 Production Configuration

**.env.production:**
```bash
DEBUG=false
LOG_LEVEL=WARNING
ALLOWED_ORIGINS=https://your-domain.com

# Use environment-specific URLs
OLLAMA_BASE_URL=http://ollama:11434
CHROMA_HOST=chromadb
REDIS_HOST=redis

# Enable authentication (implement in code)
API_KEY_ENABLED=true
API_KEY=your-secure-api-key

# Performance tuning
MAX_CONCURRENT_REQUESTS=50
REQUEST_TIMEOUT=120
```

### 9.2 SSL/TLS Configuration (Nginx)

**docker-compose.prod.yml:**
```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend
    networks:
      - codereview-network
```

**nginx.conf:**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://frontend:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 9.3 Monitoring

```bash
# Container stats
docker stats

# Prometheus metrics (implement in code)
curl http://localhost:8000/metrics
```

---

## 10. Maintenance

### 10.1 Backup

```bash
# Backup ChromaDB data
docker run --rm -v codereview-chromadb-data:/data \
  -v $(pwd)/backups:/backup \
  ubuntu tar czf /backup/chromadb-$(date +%Y%m%d).tar.gz /data

# Backup results
docker exec codereview-backend tar czf \
  /app/storage/backup-$(date +%Y%m%d).tar.gz \
  /app/storage/results
```

### 10.2 Updates

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d
```

### 10.3 Cleanup

```bash
# Remove old containers
docker compose down

# Remove volumes (WARNING: deletes data)
docker compose down -v

# Prune unused images
docker image prune -a

# Clean system
docker system prune -a
```

---

## 11. Quick Reference Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f

# Restart service
docker compose restart backend

# Enter container
docker exec -it codereview-backend bash

# Check health
curl http://localhost:8000/api/v1/health

# Pull LLM model
docker exec -it codereview-ollama ollama pull llama3:8b

# Backup data
docker run --rm -v codereview-chromadb-data:/data -v $(pwd):/backup ubuntu tar czf /backup/chromadb-backup.tar.gz /data
```

---

**Document Version:** 1.0  
**Maintained By:** DevOps Team  
**Last Updated:** February 9, 2026

**Support:** For issues, please check logs and troubleshooting section above.
