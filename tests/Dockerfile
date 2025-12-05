# Use the official Debian-hosted Python image
FROM python:3.12-slim-bookworm

ARG DEBIAN_PACKAGES="build-essential curl"

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python wants UTF-8 locale
ENV LANG=C.UTF-8

# Tell Python to disable buffering so we don't lose any logs.
ENV PYTHONUNBUFFERED=1

# Tell uv to copy packages from the wheel into the site-packages
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/home/app/.venv

# Ensure we have an up to date baseline, install dependencies
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
    mkdir -p /app && \
    chown app:app /app

# Create entrypoint script as root before switching users
RUN echo '#!/bin/bash\n\
source /home/app/.venv/bin/activate\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Switch to the new user
USER app
WORKDIR /app

# Copy dependency files first for better layer caching
# Note: Build context is project root, so use relative paths
COPY --chown=app:app ./tests/pyproject.toml ./tests/uv.lock* /app/tests/
COPY --chown=app:app ./tests/dnd_master_tests /app/tests/dnd_master_tests

# Install test dependencies using uv
WORKDIR /app/tests
RUN if [ -f uv.lock ]; then uv sync --frozen; else uv sync; fi

# Go back to app root
WORKDIR /app

# Copy source code and tests
# Build context is project root
COPY --chown=app:app ./src/ /app/src/
COPY --chown=app:app ./tests/ /app/tests/
COPY --chown=app:app ./conftest.py /app/
COPY --chown=app:app ./pytest.ini /app/

# Set Python path to include all service directories
ENV PYTHONPATH=/app/src/backend-tmp:/app/src/rule_agent:/app/src/orchestrator

# Add virtual environment to PATH so pytest and other tools are available
ENV PATH="/home/app/.venv/bin:$PATH"

# Use the entrypoint script created earlier
ENTRYPOINT ["/entrypoint.sh"]

# Default command runs all tests
CMD ["pytest", "/app/tests", "-v", "--tb=short", "--color=yes"]
