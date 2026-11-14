# Graph RAG Benchmark - Production Container Image
FROM python:3.11-slim AS runtime

# Set security and Python runtime flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=production \
    APP_HOST=0.0.0.0 \
    APP_PORT=8501

WORKDIR /app

# Install minimal OS dependencies for compilation and network health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create unprivileged application user
RUN groupadd -g 10001 graphrag && \
    useradd -u 10001 -g graphrag -s /bin/bash -m graphrag

# Copy application source code and scripts
COPY pyproject.toml /app/
COPY app.py /app/
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Create data and logs directories with proper ownership
RUN mkdir -p /app/data /app/logs && \
    chown -R graphrag:graphrag /app

# Switch to unprivileged runtime user
USER graphrag

EXPOSE 8501

# Container health probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python scripts/health_check.py || exit 1

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py"]
CMD ["--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
