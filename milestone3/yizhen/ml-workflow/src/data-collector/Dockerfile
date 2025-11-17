# =====================================================
# DND Data Collector Container
# Based on the instructor's standard MLOps container
# =====================================================

# -------- Base image --------
FROM python:3.12-slim-bookworm

# -------- System dependencies --------
ARG DEBIAN_PACKAGES="build-essential git curl wget unzip gzip"
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# -------- Install base packages and create app user --------
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install uv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && chown app:app /app && \
    mkdir -p /persistent && chown app:app /persistent && \
    mkdir -p /.venv && chown app:app /.venv

# -------- Switch to non-root user --------
USER app
WORKDIR /app

# -------- Copy dependency files --------
COPY --chown=app:app pyproject.toml uv.lock* ./

# -------- Install dependencies --------
RUN uv sync --frozen


# -------- Copy all source code --------
COPY --chown=app:app . ./

# -------- Google Cloud credentials --------
# This path matches the one in .env
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/ml-workflow.json

# -------- Default command --------
ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
