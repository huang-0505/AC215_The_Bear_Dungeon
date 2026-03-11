# 🧙‍♂️ DnD Master AI

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

**DnD Master AI** is an intelligent, AI-powered Dungeon Master system for *Dungeons & Dragons* 5th Edition adventures. Built as the capstone project for **AC215: MLOps** at Harvard University, this system provides an immersive, interactive tabletop RPG experience powered by state-of-the-art large language models and retrieval-augmented generation.

### Team Members
- Shiyu Li
- Junhui Huang
- Ruoheng Zhang
- Yizhen Wang

## Project Highlights

### 🏠 Landing Page
![Landing Page](figures/screenshot-landing.jpg)

### 📖 Narration Mode
![Narration Mode](figures/screenshot-narration.jpg)

### ⚔️ Combat Arena
![Combat Arena](figures/screenshot-combat.jpg)

- 🎮 **Interactive Web Interface** - Modern React-based UI for seamless gameplay
- ⚔️ **Turn-Based Combat System** - Full D&D 5e combat mechanics with AI-controlled enemies
- 📚 **RAG-Powered Rule Validation** - Real-time D&D rule checking using ChromaDB and semantic search
- 🧠 **Finetuned Narrator LLM** - Custom-trained Gemini model for immersive storytelling
- 🎯 **Multi-Agent Orchestration** - Intelligent routing between narration, combat, and rule validation
- 🐳 **Containerized Microservices** - Full Docker-based deployment with Nginx reverse proxy
- 🛠️ **End-to-End MLOps Pipeline** – Containerized ETL, dataset creation, and Vertex AI fine-tuning
- 📦 **DVC-Backed Data Versioning** – Deterministic dataset snapshots stored in GCS
---

## Architecture

![Project Workflow](figures/workflow.png)

### System Components

Our system consists of six core microservices:

```
┌─────────────┐
│   Frontend  │ ← Next.js 14 + TypeScript + TailwindCSS
│  (Port 80)  │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────────────┐
│            Nginx Reverse Proxy                      │
└──────┬──────────────────────────────────────────────┘
       │
┌──────▼──────────┐
│  Rule Agent     │ ← Validates player actions against D&D rules (RAG)
│  (Port 9002)    │
└──────┬──────────┘
       │ (validated action)
       │
┌──────▼──────────┐
│  Orchestrator   │ ← Main game controller with state management
│  (Port 8000)    │
└─────┬─┬─────────┘
      │ │
      │ └────────────┐
      │              │
┌─────▼──────┐  ┌───▼──────────┐  ┌──────────────┐
│  Narrator  │  │  Combat      │  │  ChromaDB    │
│  Agent     │  │  Agent       │  │  (Port 8000) │
│(Finetuned  │  │(Port 9000)   │  └──────────────┘
│  Gemini)   │  └──────────────┘
└────────────┘
```

**Request Flow:**
1. Frontend → Nginx → Rule Agent (validates action)
2. Rule Agent → Orchestrator (sends validated action)
3. Orchestrator → Narrator Agent OR Combat Agent (routes based on game state)

### Technology Stack

**Frontend:**
- ⚛️ Next.js 14 (App Router)
- 🟦 TypeScript
- 🎨 TailwindCSS + shadcn/ui
- 📦 pnpm

**Backend:**
- 🐍 Python 3.12
- ⚡ FastAPI
- 🤖 Google Gemini 2.0 Flash
- 🔍 ChromaDB (Vector Database)
- 🎲 Custom D&D 5e Combat Engine

**Infrastructure:**
- 🐳 Docker & Docker Compose
- 🌐 Nginx (Reverse Proxy)
- ☁️ Google Cloud Platform (Vertex AI)

**Model Training & Deployment:**
- 🤖 Gemini 2.5 Flash Fine-Tuning on Vertex AI Training
- 🚀 Vertex AI Endpoints for deployment of tuned narrator model

---

## Prerequisites

Before setting up the project, ensure you have:

### Required
- **Docker Desktop** (v24.0+)
  - [Install for macOS](https://docs.docker.com/desktop/install/mac-install/)
  - [Install for Windows](https://docs.docker.com/desktop/install/windows-install/)
  - [Install for Linux](https://docs.docker.com/desktop/install/linux-install/)
- **Docker Compose** (v2.0+) - Usually included with Docker Desktop
- **GCP Account** with:
  - Vertex AI API enabled
  - Gemini API access
  - Service account with appropriate permissions

### Optional (for local development)
- **Node.js** (v20+) and **pnpm** (v8+)
- **Python** (v3.11+) and **uv** or **pip**
- **Git** for version control

### GCP Setup

1. **Create a GCP Project**
   ```bash
   gcloud projects create your-project-id
   gcloud config set project your-project-id
   ```

2. **Enable Required APIs**
   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable compute.googleapis.com
   ```

3. **Create Service Account**
   ```bash
   gcloud iam service-accounts create dnd-master-sa \
       --display-name="DnD Master Service Account"

   gcloud projects add-iam-policy-binding your-project-id \
       --member="serviceAccount:dnd-master-sa@your-project-id.iam.gserviceaccount.com" \
       --role="roles/aiplatform.user"

   gcloud iam service-accounts keys create secrets/llm-service-account.json \
       --iam-account=dnd-master-sa@your-project-id.iam.gserviceaccount.com
   ```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/DaveYuan23/AC215-DnD-Master.git
cd AC215-DnD-Master
```

### 2. Configure Secrets

Create a `secrets/` directory and add your GCP credentials:

```bash
mkdir -p secrets
# Place your service account JSON key here
cp /path/to/your/credentials.json secrets/llm-service-account.json
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GCP_PROJECT=your-project-id
GCP_LOCATION=us-central1
OPENAI_API_KEY=your-openai-api-key  # Optional, for testing
```

### 4. Initialize ChromaDB (RAG Database)

Before starting the full system, initialize the D&D rules database:

```bash
# Start ChromaDB
docker compose up -d chromadb

# Build and run rule agent setup
cd src/rule_agent
chmod +x docker-shell.sh
./docker-shell.sh

# Inside the container, initialize the database
python cli.py --chunk --embed --load --chunk_type char-split

# Exit the container
exit
cd ../..
```

This process:
- Chunks the D&D 5e Player's Handbook
- Generates embeddings using Google's text-embedding-004
- Loads ~2000 rule passages into ChromaDB

---

## Deployment

### Local Deployment (Development)

#### Quick Start
```bash
# Build and start all services
docker compose up --build

# Access the application
# Frontend: http://localhost:8080
# API: http://localhost:8080/api
```

#### Service Ports
- **Frontend (Nginx)**: `8080` (external) → `3000` (internal)
- **API Gateway/Orchestrator**: `8000`
- **Combat Agent**: `9000`
- **Rule Agent**: `9002`
- **ChromaDB**: `8000` (shared with Orchestrator)

#### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f combat-agent
docker compose logs -f orchestrator
```

#### Stop Services
```bash
# Stop but keep data
docker compose stop

# Stop and remove containers
docker compose down

# Stop and remove all data (including ChromaDB)
docker compose down -v
```

### Production Deployment (Cloud Run)

For production deployment on GCP Cloud Run, see [deployment/README.md](deployment/README.md).

Key steps:
1. Build and push container images to Google Container Registry
2. Deploy each microservice to Cloud Run
3. Configure service-to-service authentication
4. Set up Cloud Load Balancer with SSL certificates
5. Configure custom domain and DNS

---

## Usage

### Starting a New Game

1. **Access the Application**
   - Navigate to `http://localhost:8080`
   - Click "Start New Adventure"

2. **Choose Your Character**
   - Select from pre-configured D&D characters:
     - **Wizard** - High intelligence spellcaster
     - **Fighter** - Martial combat specialist
     - **Rogue** - Stealthy and cunning
     - **Cleric** - Divine magic healer

3. **Select Campaign**
   - Choose from available story campaigns
   - Each campaign has unique encounters and narrative arcs

4. **Begin Your Adventure**
   - The AI Narrator will describe your starting scenario
   - Type actions in natural language
   - The system validates actions against D&D 5e rules

### Gameplay Examples

#### Narration Mode
```
Narrator: You enter a dimly lit tavern. The smell of ale and roasted meat fills the air.
         A hooded figure watches you from the corner.

You: I approach the hooded figure and ask what they want.

Narrator: The figure lowers their hood, revealing a weathered elf. "I need help,"
         they whisper. "Goblins have taken over the nearby mine..."
```

#### Combat Mode
```
⚔️ Combat Begins! Roll for Initiative!

Turn Order:
1. Wizard (You) - Initiative: 18
2. Goblin - Initiative: 15
3. Kobold - Initiative: 12

[Your Turn]
You: I cast firebolt at the goblin

Narrator: Arcane flames burst from your fingertips! The goblin shrieks as the firebolt
         strikes true, dealing 8 fire damage!

Goblin HP: 4/12
```

### API Endpoints

#### Game Management
- `POST /api/orchestrator/game/start` - Start new game session
- `POST /api/orchestrator/game/action` - Submit player action
- `GET /api/orchestrator/game/state/{session_id}` - Get current game state
- `DELETE /api/orchestrator/game/session/{session_id}` - End game session

#### Combat System
- `POST /api/combat/start` - Initialize combat session
- `GET /api/combat/state/{session_id}` - Get combat state
- `POST /api/combat/action/{session_id}` - Submit combat action
- `DELETE /api/combat/session/{session_id}` - End combat

#### Rule Validation
- `POST /api/rule-agent/validate` - Validate action against D&D rules
- `POST /api/rule-agent/retrieve_rules` - Get relevant rule passages
- `GET /api/rule-agent/health` - Service health check

For detailed API documentation, see:
- [Orchestrator API](src/orchestrator/README.md)
- [Combat Agent API](src/backend/README.md)
- [Rule Agent API](src/rule_agent/README.md)

---

## Testing

### Running Tests

The project includes comprehensive testing at three levels:

```bash
# Quick test - all levels
./run-tests.sh all

# Unit tests only (fast, isolated)
./run-tests.sh unit

# Integration tests (API testing with TestClient)
./run-tests.sh integration

# System tests (full E2E with Docker)
./run-tests.sh system

# Generate coverage report
./run-tests.sh coverage
```

### Test Structure

```
tests/
├── unit/                    # Unit tests (~50 tests)
│   └── test_combat_engine.py
├── integration/             # API integration tests (~20 tests)
│   ├── test_combat_api.py
│   ├── test_rule_agent.py
│   └── test_orchestrator.py
├── system/                  # E2E system tests (~10 tests)
│   ├── test_combat_system.py
│   └── test_full_game_flow.py
│ 
└── MLOPS/                  # E2E system tests (~10 tests)
    ├── test_data_versioning.py
    └── test_mlops_pipeline.py
```

### Continuous Integration

GitHub Actions automatically runs tests on:
- Push to `main` branch
- Pull requests to `main`

View test results in the **Actions** tab of the GitHub repository.

For detailed testing documentation, see [tests/README.md](tests/README.md).

---

## Project Structure

```
AC215-DnD-Master/
├── src/
│   ├── frontend/                   # Next.js React application
│   │   ├── app/                    # App router pages
│   │   ├── components/             # Reusable UI components
│   │   └── Dockerfile
│   │
│   ├── orchestrator/               # Main game controller
│   │   ├── app.py                  # FastAPI application
│   │   ├── game_state.py           # Dynamic campaign tree manager
│   │   ├── rule_validator.py       # Rule validation interface
│   │   └── story_trees/            # Campaign definitions
│   │
│   ├── backend/                    # Deterministic combat agent
│   │   ├── api/
│   │   │   ├── service.py          # FastAPI service
│   │   │   ├── routers/            # Combat API endpoints
│   │   │   └── utils/
│   │   │       ├── combat_engine.py    # D&D 5e combat mechanics
│   │   │       └── combat_ai.py        # Enemy AI logic
│   │   └── Dockerfile
│   │
│   ├── rule_agent/                 # RAG-based rule validation
│   │   ├── app.py                  # FastAPI application
│   │   ├── cli.py                  # Vector DB initialization
│   │   ├── agent_tools.py          # RAG pipeline implementation
│   │   └── input-datasets/         # D&D rulebooks and reference docs
│   │
│   ├── finetuning/                 # LLM finetuning workflow (legacy)
│   │   └── llm-finetuning/
│   │       ├── dataset-creator/
│   │       └── autotrain-runner/
│   │
│   ├── data-versioning/            # DVC repository for dataset snapshots
│   │   ├── data/                   # Versioned datasets
│   │   ├── data.dvc                # DVC metadata file
│   │   └── Dockerfile
│   │
│   ├── ml-workflow/                # Full MLOps pipeline (core of training system)
│   │   ├── data-collector/         # Ingest raw story logs and user interactions
│   │   │   ├── collector.py        # Data ingestion entrypoint
│   │   │   ├── config.yaml         # Collection configs
│   │   │   └── Dockerfile
│   │   │
│   │   ├── data-processor/         # Data cleaning, EDA, JSONL creation
│   │   │   ├── processor.py        # Processing pipeline
│   │   │   ├── processor_config.yaml
│   │   │   └── Dockerfile
│   │   │
│   │   ├── model-training/         # Gemini 2.5 fine-tuning pipeline
│   │   │   ├── trainer.py          # Submits Vertex AI training jobs
│   │   │   ├── training_config.yaml
│   │   │   └── Dockerfile
│   │   │
│   │   └── workflow/               # Unified workflow orchestrator
│   │       ├── workflow.py         # Calls collector → processor → trainer
│   │       └── config.yaml
│   │
│   ├── infra/
│   │   └── nginx.conf              # Reverse proxy config
│   │
│   └── rag/
│       └── docker-volumes/         # Persistent ChromaDB data
│
├── tests/                          # Comprehensive multi-stage test suite
│   ├── unit/
│   ├── integration/
│   └── system/
│   └── MLOPS/
│
├── deployment/                     # Cloud deployment configs (GCP, Vertex AI)
├── figures/                        # Documentation images
├── notebooks/                      # EDA and model experiment notebooks
├── docker-compose.yml              # Multi-service orchestration
├── .github/workflows/              # CI/CD pipelines for linting, tests, builds
└── README.md

```

---

## Features in Detail

### 🎮 Interactive Narration
- **AI-Generated Stories**: Finetuned Gemini 2.0 model creates immersive narratives
- **Dynamic World**: Environment responds to player choices
- **Multiple Campaigns**: Pre-built story arcs with branching paths
- **State Management**: Full game history tracking with tree structure

### ⚔️ Combat System
- **D&D 5e Mechanics**: Initiative rolls, attack rolls, damage calculation
- **AI Enemies**: Gemini-powered tactical decision making
- **Smart Targeting**: Enemies prioritize wounded targets and tactical positioning
- **Dramatic Narration**: Every action gets cinematic description
- **Turn-Based Flow**: Clear indication of turn order and combat state

### 📚 Rule Validation
- **RAG Pipeline**: Semantic search over D&D 5e Player's Handbook
- **Real-Time Validation**: Actions checked against official rules
- **Helpful Corrections**: Suggests valid alternatives for invalid actions
- **Sabotage Detection**: Prevents meta-gaming and rule-breaking
- **Context-Aware**: Validation considers current game state

### 🧠 Multi-Agent Architecture
- **Intelligent Routing**: Automatic switching between narration and combat
- **State Transitions**: Seamless mode changes based on game events
- **Session Management**: Persistent game state across interactions
- **Error Resilience**: Graceful degradation when services unavailable

### 🛠️ MLOps Pipeline**
- **Fully Containerized Workflow:** Data collection, processing, and training packaged into reproducible Docker images
- **DVC Versioning**: All datasets and training artifacts tracked with deterministic hashes and stored in GCS
- **Automated Dataset Builder**: ETL pipeline converts gameplay logs into clean, structured JSONL suitable for fine-tuning
- **Config-Driven Training**: Vertex AI training jobs launched through YAML-defined parameters ensuring consistent runs
- **Model Lifecycle Management**: Versioned fine-tuned models deployed to Vertex Endpoints with rollback capability

---

## Known Issues and Limitations

### Current Limitations

1. **In-Memory Session Storage**
   - Game sessions stored in memory, not persistent across restarts
   - **Workaround**: Don't restart services during active games
   - **Future**: Migrate to Redis or database storage

2. **Single-Player Only**
   - Currently supports one player with AI teammates
   - **Future**: Implement multiplayer with WebSocket support

3. **Limited Campaign Library**
   - Few pre-built campaigns available
   - **Workaround**: Campaigns can be added via JSON in `src/orchestrator/story_trees/`

4. **Combat Complexity**
   - Advanced D&D features not yet implemented (spell slots, concentration, etc.)
   - **Workaround**: System handles most common combat scenarios

5. **LLM Response Times**
   - Occasional delays (2-3 seconds) for narrator and enemy AI
   - **Workaround**: Frontend shows loading indicators

### Known Bugs

- **Enemy First Turn**: Rare race condition if enemy has very high initiative (fixed with 3s delay and retry logic)
- **ChromaDB Initialization**: First-time setup requires manual CLI run
- **Port Conflicts**: If ports 8080, 8000, 9000, or 9002 are in use, services won't start

### Reporting Issues

Found a bug? Please [open an issue](https://github.com/DaveYuan23/AC215-DnD-Master/issues) with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Console logs (if applicable)

---

## Development

### Local Development Setup

#### Frontend Development
```bash
cd src/frontend

# Install dependencies
pnpm install

# Run dev server (hot reload)
pnpm dev

# Access at http://localhost:3000
```

#### Backend Development
```bash
cd src/backend

# Enter development container
./docker-shell.sh

# Inside container
uvicorn api.service:app --reload --host 0.0.0.0 --port 9000
```

#### Rule Agent Development
```bash
cd src/rule_agent

# Enter development container
./docker-shell.sh

# Inside container
uvicorn app:app --reload --host 0.0.0.0 --port 9002

```

#### DVC Development
```bash
# Initialize and Configure DVC (first time)
cd src/data-versioning

## Initialize DVC in this directory
dvc init

## Add GCS remote
dvc remote add -d gcs_remote gs://dnd-data-versioning

# Pull Versioned Data
## Download the latest versioned dataset from GCS
dvc pull

# Push Updated Data
## After modifying data/ directory
dvc add data
git add data.dvc
git commit -m "Update dataset"

## Upload tracked dataset to GCS
dvc push

# Check Dataset Version History
##  See tracked dataset versions through Git
git log -- data.dvc
```

#### MLOps Development
```bash
cd src/ml-workflow/workflow

# Run collector → processor → trainer in sequence
python workflow.py --config config.yaml
```

### Code Quality Tools

```bash
# Format Python code
black src/

# Sort imports
isort src/

# Lint Python code
flake8 src/

# Type checking
mypy src/
```

### Adding New Features

1. **New Campaign**: Add JSON to `src/orchestrator/story_trees/`
2. **New Combat Action**: Extend `ACTION_REGISTRY` in `combat_engine.py`
3. **New Agent**: Create service, add to `docker-compose.yml`, update orchestrator routing
4. **New Rule Check**: Modify `SYSTEM_INSTRUCTION` in `rule_agent/cli.py`

---

## Documentation

### Detailed Module Documentation
- [Combat Agent](src/backend/README.md) - Combat mechanics and AI
- [Orchestrator](src/orchestrator/README.md) - Game state management
- [Rule Agent](src/rule_agent/README.md) - RAG pipeline
- [Frontend](src/frontend/README.md) - UI components
- [Testing](tests/README.md) - Test suite documentation

### Feature Guides
- [Story Tree System](STORY_TREE_GUIDE.md) - Campaign creation
- [Round-Based Combat](ROUND_BASED_COMBAT_GUIDE.md) - Combat mechanics
- [Campaign Integration](CAMPAIGN_INTEGRATION_GUIDE.md) - Adding campaigns

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **AC215 Course Staff** - Harvard University
- **D&D 5e System Reference Document** - Wizards of the Coast
- **Google Cloud Platform** - Vertex AI and Gemini models
- **Open Source Community** - FastAPI, Next.js, ChromaDB, and all dependencies

---

## Contact

For questions or collaboration:
- **GitHub Issues**: [Create an issue](https://github.com/DaveYuan23/AC215-DnD-Master/issues)
- **Course**: AC215 - MLOps, Harvard University

---

**Built with ❤️ for the love of D&D and AI**
