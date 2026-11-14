# Phase 9 Production Deployment Guide

**Project:** Graph RAG Benchmark  
**Current Version:** `0.8.0`  
**Target Version:** `0.9.0`  
**Phase:** 9 — Production Readiness & Deployment  
**Author:** DevSecOps Specialist & Cloud Deployment Architect  

---

## 1. Deployment Profiles Overview

The **Graph RAG Benchmark** application supports two primary deployment topologies:
- **Profile A (Local Windows Deployment):** High-performance workstation deployment utilizing native hardware acceleration (NVIDIA RTX GPU) via host-managed Ollama.
- **Profile B (Containerized Single-Host Deployment):** Docker Compose deployment isolating application runtime and dependencies while communicating with host-managed Ollama via `host.docker.internal`.

---

## 2. Profile A: Windows Local Deployment

### 2.1 Prerequisites
- Windows 11 / Windows Server 2022
- Python 3.11.9 64-bit
- Ollama installed natively on Windows with models pulled:
  ```powershell
  ollama pull qwen3:8b
  ollama pull nomic-embed-text
  ```

### 2.2 Environment Setup & Installation
```powershell
# 1. Clone / Navigate to workspace root
cd "C:\VS CODE\Graph RAG Benchmark"

# 2. Create isolated virtual environment (if not already created)
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Configuration
Copy `.env.example` to `.env` and verify settings:
```powershell
Copy-Item .env.example .env
```

### 2.4 Pre-Flight Verification & Health Check
```powershell
.\.venv\Scripts\python.exe scripts/health_check.py
```
Expected output:
```text
Health Status: HEALTHY
Liveness:      OK
Readiness:     OK
Summary:       All 13 production subsystems are healthy and ready for traffic.
```

### 2.5 Launching the Application
Execute the production launcher script:
```powershell
.\scripts\start_production.ps1
```
The application will launch on `http://127.0.0.1:8501`.

### 2.6 Graceful Stop Procedure
Press `Ctrl + C` in the running PowerShell terminal to terminate the Streamlit server. Active benchmark results are written atomically to disk and are preserved.

---

## 3. Profile B: Containerized Docker Deployment

### 3.1 Prerequisites
- Docker Engine $\ge 24.0$ and Docker Compose $\ge 2.20$ (e.g. Docker Desktop on Windows).
- Host Ollama running on port 11434 with `OLLAMA_ORIGINS=*` or accepting requests from Docker network.

### 3.2 Building the Container Image
```bash
docker compose build
```

### 3.3 Launching the Container Service
```bash
docker compose up -d
```

### 3.4 Monitoring Container Health & Logs
```bash
# Check container status and health probe
docker compose ps

# Follow container structured logs
docker compose logs -f graphrag-bench
```

### 3.5 Graceful Shutdown
```bash
docker compose down
```

### 3.6 Upgrading & Rollback Procedures
```bash
# Upgrade: Pull/build new image, verify health, and recreate container
docker compose build --no-cache
docker compose up -d --force-recreate

# Rollback: If new version exhibits regression, restore previous image and data
docker compose down
python scripts/restore.py --restore <pre_upgrade_backup_id> --confirm
docker compose up -d
```

---

## 4. Secure Reverse Proxy Configuration (TLS & Authentication)

For remote network access, Streamlit **must** sit behind a TLS reverse proxy (e.g. Nginx or Caddy) enforcing authentication and security headers.

### Sample Nginx Configuration (`/etc/nginx/conf.d/graphrag.conf`):
```nginx
server {
    listen 443 ssl http2;
    server_name rag.internal.corp;

    ssl_certificate     /etc/ssl/certs/graphrag.crt;
    ssl_certificate_key /etc/ssl/private/graphrag.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # Basic Authentication (or forward to OAuth2-Proxy / Authelia)
    auth_basic "Graph RAG Benchmark Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    # Security Headers
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options SAMEORIGIN always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;" always;

    # Maximum file upload size matching application config
    client_max_body_size 25M;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
    }
}
```
