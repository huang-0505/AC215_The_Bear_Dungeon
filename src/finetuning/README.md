# 🧙‍♂️ QLoRA Fine-tuning for DND Narrator (AC215 Project)

This container fine-tunes a **Narrator-style LLM** using **QLoRA (4-bit LoRA adaptation)**.  
It is built for GPU-based local training inside a Docker container, following the AC215 project structure.

---

## 📂 Project Structure
finetuning/
├── Dockerfile # GPU base image (PyTorch + CUDA) with uv dependency management
├── docker-entrypoint.sh # Default container entrypoint (runs training automatically)
├── docker-shell.sh # Local helper script: build + run container interactively
├── pyproject.toml # Dependency list (transformers, peft, trl, etc.)
├── uv.lock # Auto-generated dependency lock file
├── cli.py # Main training / chat CLI for QLoRA fine-tuning
├── .dockerignore # Exclude large or irrelevant files from the image
└── README.md # This file