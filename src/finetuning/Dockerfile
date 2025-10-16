# syntax=docker/dockerfile:1

# --- GPU base image with PyTorch + CUDA + cuDNN ---
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# --- Environment setup ---
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/.venv \
    HF_HOME=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf

# --- Install system dependencies and tini ---
RUN set -ex; \
    apt-get update && \
    apt-get install -y --no-install-recommends git curl ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

# --- Install uv package manager ---
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

# --- Create a non-root user and working directories ---
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} -s /bin/bash app && \
    mkdir -p /app /.venv /data /outputs ${HF_HOME} && \
    chown -R app:app /app /.venv /data /outputs ${HF_HOME}

USER app
WORKDIR /app

# --- Add venv to PATH ---
ENV PATH="/.venv/bin:${PATH}"

# --- Copy dependency files first for caching ---
COPY --chown=app:app pyproject.toml uv.lock* ./

# --- Install dependencies (create venv if not yet created) ---
RUN uv sync || true

# --- Copy source files ---
COPY --chown=app:app cli.py ./cli.py
COPY --chown=app:app docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# --- Default training parameters (can be overridden by -e flags) ---
ENV MODEL_NAME="mistralai/Mistral-7B-v0.1" \
    DATA_DIR="/data" \
    OUTPUT_DIR="/outputs" \
    MAX_STEPS=200 \
    PER_DEVICE_TRAIN_BATCH_SIZE=1 \
    GRADIENT_ACCUMULATION_STEPS=8 \
    LR=2e-4

# --- Entrypoint and CMD ---
ENTRYPOINT ["/usr/bin/tini","--","docker-entrypoint.sh"]
CMD ["python","cli.py","--model_name","${MODEL_NAME}","--data_dir","${DATA_DIR}","--output_dir","${OUTPUT_DIR}","--max_steps","${MAX_STEPS}","--per_device_train_batch_size","${PER_DEVICE_TRAIN_BATCH_SIZE}","--gradient_accumulation_steps","${GRADIENT_ACCUMULATION_STEPS}","--learning_rate","${LR}"]
