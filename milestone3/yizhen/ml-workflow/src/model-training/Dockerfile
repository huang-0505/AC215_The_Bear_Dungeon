# ============================================================
# Model Training Container for AC215 DnD Project
# ============================================================

FROM python:3.12-slim-bookworm

# ------------------------------------------------------------
# Basic environment setup
# ------------------------------------------------------------

ARG DEBIAN_PACKAGES="build-essential git curl wget unzip gzip"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

# uv settings
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

# ------------------------------------------------------------
# Install system dependencies + uv + create user + dirs
# ------------------------------------------------------------
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    pip install --no-cache-dir --upgrade pip && \
    pip install uv && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    useradd -ms /bin/bash app -d /home/app -u 1000 && \
    mkdir -p /app && chown app:app /app && \
    mkdir -p /persistent && chown app:app /persistent && \
    mkdir -p /.venv && chown app:app /.venv

# Switch user and set working directory
USER app
WORKDIR /app

# ------------------------------------------------------------
# Install Python dependencies with uv
# ------------------------------------------------------------

# Only copy dependency declarations first for clean layer caching
COPY --chown=app:app pyproject.toml uv.lock* ./

RUN uv sync --frozen

# ------------------------------------------------------------
# Copy the rest of the project
# ------------------------------------------------------------
COPY --chown=app:app . ./

# ------------------------------------------------------------
# Entry point → handled by docker-entrypoint.sh
# ------------------------------------------------------------
ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
