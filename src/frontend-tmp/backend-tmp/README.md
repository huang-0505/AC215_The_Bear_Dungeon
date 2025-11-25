# DnD Combat Backend API

FastAPI-based backend service for the DnD Combat Simulator, integrating the combat agent with AI-powered narration.

## Features

- **Combat Engine**: Turn-based combat system with initiative rolls, actions, and HP tracking
- **AI Enemy Decisions**: Google GenAI-powered enemy AI that makes tactical combat decisions
- **AI Narration**: Generates vivid, dramatic D&D-style combat narratives
- **RESTful API**: Clean endpoints for combat session management
- **Semantic Action Parsing**: Natural language action understanding (optional with embeddings)

## API Endpoints

### `POST /combat/start`
Start a new combat session with preset or custom characters.

**Request Body** (optional):
```json
{
  "players": [...],
  "enemies": [...]
}
```

**Response**:
```json
{
  "session_id": "uuid",
  "message": "Combat initiated!",
  "state": { ... }
}
```

### `GET /combat/state/{session_id}`
Get current combat state.

### `POST /combat/action/{session_id}`
Submit a player action.

**Request Body**:
```json
{
  "action": "I swing my sword at the goblin"
}
```

**Response**:
```json
{
  "narrative": "With a mighty swing...",
  "raw_result": "⚔️ Knight slashes Goblin for 12 damage!",
  "state": { ... }
}
```

### `DELETE /combat/session/{session_id}`
End a combat session.

## Running the Backend

### Prerequisites

- Docker
- GCP credentials with Vertex AI access
- GCP project with Vertex AI API enabled

### Setup

1. Place your GCP credentials in `../../../secrets/llm-service-account.json`

2. Update `docker-shell.sh` with your GCP project:
   ```bash
   export GCP_PROJECT="your-project-id"
   ```

3. Run the backend:
   ```bash
   chmod +x docker-shell.sh
   ./docker-shell.sh
   ```

4. Inside the container:
   ```bash
   uvicorn_server
   ```

The API will be available at `http://localhost:9000`

## Project Structure

```
backend-tmp/
├── api/
│   ├── service.py              # FastAPI application
│   ├── routers/
│   │   └── combat.py           # Combat endpoints
│   └── utils/
│       ├── combat_engine.py    # Combat logic
│       ├── combat_ai.py        # AI components
│       └── db_tool.py          # Embedding utilities
├── Dockerfile
├── pyproject.toml
├── docker-entrypoint.sh
└── docker-shell.sh
```

## Environment Variables

- `GCP_PROJECT`: Your GCP project ID
- `GCP_LOCATION`: GCP region (default: us-central1)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account key

## Combat Flow

1. Client starts combat → Creates session with default or custom characters
2. Engine rolls initiative → Determines turn order
3. For each turn:
   - If player: Accept natural language action → Parse → Execute
   - If enemy: AI decides action → Execute
4. Generate narrative for each action
5. Update character states
6. Check for battle end condition
7. Return state to client

## AI Models Used

- **Enemy AI**: `gemini-2.0-flash-001` for tactical decision-making
- **Narrator**: `gemini-2.0-flash-001` for combat narration
- **Embeddings** (optional): `text-embedding-004` for semantic action parsing

## Development

To modify combat rules, edit [combat_engine.py](api/utils/combat_engine.py).

To adjust AI behavior, edit [combat_ai.py](api/utils/combat_ai.py).

To add new endpoints, create routers in `api/routers/` and register them in [service.py](api/service.py).
