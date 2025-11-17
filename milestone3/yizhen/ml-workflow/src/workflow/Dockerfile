# ============================================================
# Base image
# ============================================================
FROM python:3.11-slim-bullseye

# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# ------------------------------------------------------------
# Install system dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    ca-certificates \
    gnupg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install uv (Python package manager)
# ------------------------------------------------------------
RUN pip install --no-cache-dir uv

# ------------------------------------------------------------
# Create runtime directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# Copy project metadata (pyproject.toml + uv.lock)
# MUST be copied before source code for layer caching
# ------------------------------------------------------------
COPY pyproject.toml uv.lock* ./

# ------------------------------------------------------------
# Install dependencies into /.venv
# ------------------------------------------------------------
RUN uv sync --frozen

# ------------------------------------------------------------
# Copy the full project
# ------------------------------------------------------------
COPY . .

# ------------------------------------------------------------
# Entrypoint (all commands go through docker-entrypoint.sh)
# ------------------------------------------------------------
ENTRYPOINT ["bash", "docker-entrypoint.sh"]
