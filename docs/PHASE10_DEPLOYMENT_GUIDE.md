# Graph RAG Benchmark — Phase 10 Deployment & Infrastructure Guide

**Document Version:** `1.0.0`  
**Release Target:** `v0.10.0` (Phase 10 — Real-User Pilot Deployment & Product Launch)  
**Classification:** DevOps / SRE Infrastructure Runbook  
**Status:** Approved for Deployment  

---

## 1. Architectural Overview: Controlled Single-Host Topology

For Phase 10, Graph RAG Benchmark is designed to run in a **hardened, single-host pilot deployment topology**. All computationally intensive components—Ollama LLM engine, ChromaDB vector store, NetworkX knowledge graph, and Streamlit web application—operate on a single physical or virtual host.

Internet and corporate intranet traffic is terminated at a **TLS Reverse Proxy** (Nginx or Caddy) which forwards authenticated traffic over loopback (`127.0.0.1`) to the Streamlit pilot application.

```
+-----------------------------------------------------------------------------------+
|                        PILOT DEPLOYMENT INFRASTRUCTURE                             |
+-----------------------------------------------------------------------------------+

[ Pilot Evaluators ]
       |
       | HTTPS (Port 443) / WSS (WebSocket)
       v
+-----------------------------------------------------------------------------------+
| Host: Local Server / Workstation (Windows 11 / Linux)                             |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | TLS Reverse Proxy (Nginx / Caddy)                                           |  |
|  |  • TLS 1.3 Termination (Valid Certificate)                                  |  |
|  |  • WebSocket Upgrade Headers (Connection: Upgrade)                         |  |
|  |  • HSTS, CSP, X-Frame-Options Security Headers                              |  |
|  +-----------------------------------------------------------------------------+  |
|         |                                                                         |
|         | HTTP (127.0.0.1:8501)                                                   |
|         v                                                                         |
|  +-----------------------------------------------------------------------------+  |
|  | Streamlit Pilot Web Application (v0.10.0)                                   |  |
|  |  • Native OIDC Authentication Bridge                                        |  |
|  |  • Role-Based Access Control (Admin, Researcher, Viewer)                    |  |
|  |  • Inactivity Session Timer (120 min timeout)                               |  |
|  |  • Attributable Audit Stream (`logs/user_audit.jsonl`)                      |  |
|  +-----------------------------------------------------------------------------+  |
|         |                                           |                             |
|         | In-Process Library                        | HTTP (127.0.0.1:11434)      |
|         v                                           v                             |
|  +-------------------------------------+   +-----------------------------------+  |
|  | Local Storage & Indexes             |   | Ollama Local AI Engine            |  |
|  |  • ChromaDB (`data/chroma/`)        |   |  • Qwen3:8b (Grounded QA & KG)    |  |
|  |  • NetworkX Graph (`data/kg/`)      |   |  • nomic-embed-text (768-dim)     |  |
|  |  • Corpus Repo (`data/processed/`)  |   |  • NVIDIA RTX 3050 (4 GB VRAM)    |  |
|  |  • User Registry (`data/users/`)    |   +-----------------------------------+  |
|  +-------------------------------------+                                          |
+-----------------------------------------------------------------------------------+
```

---

## 2. Reverse Proxy TLS Configurations

Because Streamlit relies heavily on persistent WebSockets (`_stcore/stream`), reverse proxies must explicitly configure connection upgrade headers and long read timeouts.

### 2.1 Production Nginx Configuration Snippet
```nginx
server {
    listen 443 ssl http2;
    server_name graphrag-pilot.yourdomain.com;

    # SSL Certificates
    ssl_certificate /etc/letsencrypt/live/graphrag-pilot.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/graphrag-pilot.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Enterprise Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Forward to Streamlit Pilot
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;

        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Standard proxy headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable proxy buffering for streaming LLM tokens
        proxy_buffering off;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
}
```

### 2.2 Caddy Alternative Configuration (Auto-HTTPS)
```caddy
graphrag-pilot.yourdomain.com {
    reverse_proxy 127.0.0.1:8501 {
        header_up Host {host}
        header_up X-Real-IP {remote}
    }
}
```

---

## 3. Step-by-Step Pilot Deployment on Windows

### Step 1: Clone Repository & Setup Virtual Environment
```powershell
Set-Location "C:\VS CODE\Graph RAG Benchmark"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### Step 2: Verify Ollama Engine & Models
Ensure Ollama is running and has both required models loaded:
```powershell
ollama list
# Ensure qwen3:8b and nomic-embed-text are present
```

### Step 3: Configure Authentication Secrets
1. Copy the template secrets file:
   ```powershell
   Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
   ```
2. Open `.streamlit\secrets.toml` and configure your enterprise OIDC parameters:
   - `client_id`
   - `client_secret`
   - `server_metadata_url`
   - `redirect_uri` (e.g. `https://graphrag-pilot.yourdomain.com/oauth2callback`)

### Step 4: Configure Admin Bootstrap in `.env`
Create or edit `.env`:
```bash
GRAPHRAG_AUTH_MODE=oidc
GRAPHRAG_ADMIN_BOOTSTRAP_EMAIL=lead.evaluator@yourdomain.com
GRAPHRAG_SESSION_TIMEOUT_MINUTES=120
MAX_CONCURRENT_GENERATIONS=1
```

### Step 5: Run Pre-Flight Validation
```powershell
python scripts/check_auth_config.py
python scripts/pilot_health_check.py
```

### Step 6: Launch the Pilot Server
```powershell
.\scripts\start_pilot.ps1 -AuthMode oidc -Port 8501 -NoBrowser
```
The script records the process PID in `logs/pilot.pid` and confirms startup.

### Step 7: Clean Shutdown
To stop the pilot server cleanly:
```powershell
.\scripts\stop_pilot.ps1 -Port 8501
```

---

## 4. Containerized Docker Deployment

For Linux server hosts or containerized clusters, use Docker:

```dockerfile
# Multi-stage secure build
FROM python:3.11-slim AS runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app.py ./
COPY scripts/ scripts/

ENV PYTHONPATH=/app/src
ENV GRAPHRAG_AUTH_MODE=oidc

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Run with Docker Compose
```yaml
version: "3.8"
services:
  pilot-app:
    build: .
    ports:
      - "127.0.0.1:8501:8501"
    environment:
      - GRAPHRAG_AUTH_MODE=oidc
      - GRAPHRAG_ADMIN_BOOTSTRAP_EMAIL=lead.evaluator@yourdomain.com
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro
    restart: unless-stopped
```
