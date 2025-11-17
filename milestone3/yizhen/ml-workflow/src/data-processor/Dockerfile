# ============================================================
# Data Processor Container for AC215 DnD Project
# ============================================================

# Use the official Debian-hosted Python image
FROM python:3.12-slim-bookworm

# ------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------
ARG DEBIAN_PACKAGES="build-essential git curl wget unzip gzip"

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive
# Python wants UTF-8 locale
ENV LANG=C.UTF-8
# Disable stdout buffering
ENV PYTHONUNBUFFERED=1
# uv settings
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# ------------------------------------------------------------
# Install system dependencies, pip, and uv
# ------------------------------------------------------------
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install uv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && chown app:app /app && \
    mkdir -p /persistent && chown app:app /persistent && \
    mkdir -p /.venv && chown app:app /.venv

# ------------------------------------------------------------
# Switch to non-root user and set workdir
# ------------------------------------------------------------
USER app
WORKDIR /app

# ------------------------------------------------------------
# Install Python dependencies via uv
# ------------------------------------------------------------
COPY --chown=app:app pyproject.toml uv.lock* ./
RUN uv sync --frozen


# ------------------------------------------------------------
# Copy source code
# ------------------------------------------------------------
COPY --chown=app:app . ./

# ------------------------------------------------------------
# Entry point (to be defined in docker-entrypoint.sh)
# ------------------------------------------------------------
ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
