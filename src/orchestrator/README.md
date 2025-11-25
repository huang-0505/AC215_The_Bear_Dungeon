# D&D Game Orchestrator Architecture

## Overview

The enhanced orchestrator implements a complete game state management system with D&D rule validation, agent routing, and automatic state transitions between narration and combat modes.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Input                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Port 8000)                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Build Game Context (from State Tree)                │   │
│  │  2. Validate with Rule Agent                            │   │
│  │  3. Check for Sabotage/Invalid Actions                  │   │
│  │  4. Route to Appropriate Agent                          │   │
│  │  5. Detect State Transitions                            │   │
│  │  6. Update Game State Tree                              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────┬────────────────┬────────────────┬───────────────────────┘
      │                │                │
      ↓                ↓                ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Rule Agent   │ │ Narrator     │ │ Combat Agent │
│ (Port 9002)  │ │ (Port 9001)  │ │ (Port 9000)  │
│              │ │              │ │              │
│ - ChromaDB   │ │ - Finetuned  │ │ - Turn-based │
│ - RAG        │ │   LLM        │ │   combat     │
│ - D&D Rules  │ │ - Story gen  │ │ - AI enemies │
└──────────────┘ └──────────────┘ └──────────────┘
```

## Components

### 1. Game State Tree ([game_state.py](src/orchestrator/game_state.py))

**Purpose**: Manages the complete game history as a tree structure.

**Key Classes**:
- `GameStateType`: Enum for state types (NARRATION, COMBAT, DIALOGUE, etc.)
- `AgentType`: Enum for agent types (NARRATOR, COMBAT, ORCHESTRATOR)
- `GameStateNode`: Individual state with metadata, actions, responses, validation results
- `GameStateTree`: Tree manager with traversal and transition methods

**Features**:
- Full history tracking
- Parent-child relationships
- Rule validation storage per node
- State transition detection

**Example**:
```python
tree = GameStateTree()
root = tree.create_root(GameStateType.NARRATION)
root.narrative_text = "You enter a tavern..."

# Later, when combat triggers
combat_node = tree.add_child(
    parent_id=root.id,
    state_type=GameStateType.COMBAT,
    agent=AgentType.COMBAT
)
tree.transition_to(combat_node.id)
```

### 2. Rule Validator ([rule_validator.py](src/orchestrator/rule_validator.py))

**Purpose**: Interface to the Rule Agent for D&D rules validation.

**Methods**:
- `validate_action(user_input, game_context)`: Full validation with RAG retrieval
- `is_sabotage(validation_result)`: Detect meta-gaming attempts
- `get_applicable_rules(action)`: Retrieve relevant D&D rules
- `check_health()`: Service availability check

**Returns**:
```json
{
  "is_valid": true,
  "validation_type": "valid",
  "explanation": "According to D&D rules...",
  "rule_text": "Attack Action: When you take the Attack action...",
  "suggested_correction": null
}
```

### 3. Context Builder ([context_builder.py](src/orchestrator/context_builder.py))

**Purpose**: Extract game context from the state tree for validation and agent calls.

**Methods**:
- `build_context(tree)`: Extract current state, recent actions, combat info
- `build_agent_context(node, tree)`: Rich context for agent calls
- `get_story_summary(tree)`: Generate narrative summary

**Context Example**:
```python
{
  "state_type": "combat",
  "combat_session_id": "uuid-123",
  "in_combat": true,
  "recent_actions": [
    {"action": "I attack the goblin", "response": "You strike..."}
  ]
}
```

### 4. Rule Agent API ([rule_agent/app.py](src/rule_agent/app.py))

**Purpose**: FastAPI wrapper for the D&D RAG system.

**Endpoints**:
- `POST /validate`: Validate action against D&D rules
- `POST /retrieve_rules`: Get relevant rule passages
- `GET /health`: Service health check

**Features**:
- ChromaDB vector database integration
- Gemini LLM with function calling
- Sabotage pattern detection
- Graceful fallback when DB unavailable

### 5. Enhanced Orchestrator ([orchestrator/app.py](src/orchestrator/app.py))

**Purpose**: Main game controller with complete validation pipeline.

**New Endpoints**:

#### `POST /game/start`
Initialize a new game session.
```json
{
  "initial_prompt": "Start a new D&D adventure in a dark forest."
}
```

**Response**:
```json
{
  "session_id": "uuid-456",
  "response": "You find yourself in a dark forest...",
  "state": { "id": "node-1", "state_type": "narration", ... }
}
```

#### `POST /game/action`
Submit player action with full validation.
```json
{
  "session_id": "uuid-456",
  "text": "I attack the goblin with my sword"
}
```

**Response**:
```json
{
  "session_id": "uuid-456",
  "state_type": "combat",
  "agent_used": "combat",
  "response": "⚔️ Combat begins!",
  "validation": {
    "is_valid": true,
    "rule_text": "Attack Action: When you take the Attack action...",
    "explanation": "This is a valid melee attack..."
  },
  "state_node": { ... },
  "transition": "narration -> combat"
}
```

#### `GET /game/state/{session_id}`
Get complete game state and history.

#### `DELETE /game/session/{session_id}`
End game session.

## Request Flow

### Starting a Game

```
1. Client → POST /game/start
2. Orchestrator creates GameStateTree with root node
3. Orchestrator → Narrator Agent: Generate initial story
4. Orchestrator checks for combat trigger
5. Orchestrator updates tree and returns response
```

### Player Action Flow

```
1. Client → POST /game/action {"text": "I attack the goblin"}
2. Orchestrator builds game context from tree
3. Orchestrator → Rule Agent: Validate action
   ↓
4a. If sabotage detected → Return error to client
4b. If valid → Continue
   ↓
5. Store validation in current node
6. Route to agent based on current state:

   If NARRATION:
   7a. Orchestrator → Narrator Agent
   8a. Check response for combat trigger
   9a. If combat triggered:
       - Create combat child node
       - Orchestrator → Combat Agent: Start combat
       - Transition to combat node
   10a. Return response

   If COMBAT:
   7b. Orchestrator → Combat Agent: Submit action
   8b. Check for combat end
   9b. If combat ended:
       - Create narration child node
       - Orchestrator → Narrator Agent: Continue story
       - Transition to narration node
   10b. Return response
```

## State Transitions

### Narration → Combat
**Trigger**: LLM detects combat keywords in narrator response
- "enemies appear", "you are attacked", "roll for initiative"

**Actions**:
1. Create combat child node
2. Store combat trigger info
3. Start combat session via Combat Agent
4. Transition tree to combat node

### Combat → Narration
**Trigger**: Combat state indicates battle_over = true

**Actions**:
1. Create narration child node
2. Store combat outcome (winner)
3. Generate post-combat narration
4. Transition tree to narration node

## Validation Pipeline

### 1. Rule Validation
```python
validation = rule_validator.validate_action(user_input, game_context)
```

### 2. Sabotage Detection
```python
if rule_validator.is_sabotage(validation):
    return error_response("Meta-gaming detected")
```

### 3. Rule Storage
```python
current_node.rule_validation = validation
current_node.applicable_rules = validation["rule_text"]
```

### 4. Rule Context Passing
```python
# Pass rules to agents for context-aware responses
narrator_response = call_narrator_agent(
    user_input,
    rules_context=validation["rule_text"]
)
```

## Service Configuration

### Environment Variables

**Orchestrator**:
```bash
OPENAI_API_KEY=sk-...
RULE_AGENT_URL=http://rule-agent:9002
COMBAT_AGENT_URL=http://combat-agent:9000
NARRATOR_AGENT_URL=http://narrator-agent:9001
```

**Rule Agent**:
```bash
GCP_PROJECT=your-project-id
GCP_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/secrets/llm-service-account.json
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000
```

**Combat Agent**:
```bash
GCP_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/secrets/llm-service-account.json
```

## Running the System

### 1. Setup ChromaDB (Rule Agent prerequisite)
```bash
cd src/rule_agent
./docker-shell.sh

# Inside container
python cli.py --chunk --embed --load --chunk_type char-split
```

### 2. Start All Services
```bash
# From project root
docker-compose up --build
```

### 3. Test the Orchestrator
```bash
# Start a game
curl -X POST http://localhost/api/game/start \
  -H "Content-Type: application/json" \
  -d '{"initial_prompt": "Start a D&D adventure"}'

# Submit an action
curl -X POST http://localhost/api/game/action \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "text": "I attack the goblin with my sword"
  }'
```

## Game State Tree Example

```
Root (Narration)
├── id: node-1
├── narrative_text: "You enter a tavern..."
├── player_action: "Start adventure"
└── children: [node-2]
    │
    ├─→ Combat Node
    │   ├── id: node-2
    │   ├── combat_session_id: combat-uuid-123
    │   ├── player_action: "I draw my sword"
    │   ├── rule_validation: {is_valid: true, ...}
    │   └── children: [node-3]
    │       │
    │       ├─→ Narration Node (Post-Combat)
    │           ├── id: node-3
    │           ├── narrative_text: "Victory! The goblins flee..."
    │           └── metadata: {combat_outcome: "players"}
```

## Extending the System

### Adding a New Agent

1. Create agent service (e.g., `narrator_agent/`)
2. Add to `docker-compose.yml`
3. Update orchestrator agent URLs
4. Add agent call function in `orchestrator/app.py`
5. Update routing logic if needed

### Adding a New State Type

1. Add to `GameStateType` enum in `game_state.py`
2. Create handler function in `orchestrator/app.py`
3. Update routing logic in `game_action()`
4. Add transition detection if needed

### Customizing Rule Validation

1. Modify `SYSTEM_INSTRUCTION` in `rule_agent/cli.py`
2. Update sabotage keywords in `rule_agent/app.py`
3. Adjust validation logic in `validate_action()`

## Key Design Decisions

1. **In-Memory Session Storage**: Currently uses dict, should migrate to Redis for production
2. **Rule Agent is Informative**: Validates but doesn't block actions (can be changed)
3. **LLM-Based Combat Detection**: Uses GPT-4o-mini for state transition detection
4. **Graceful Degradation**: Services work even if Rule Agent is unavailable
5. **Backward Compatibility**: Legacy `/orchestrate` endpoint preserved

## Next Steps

- [ ] Implement Narrator Agent API wrapper for finetuned model
- [ ] Add Redis for session persistence
- [ ] Implement save/load game functionality
- [ ] Add WebSocket support for real-time updates
- [ ] Create frontend integration
- [ ] Add comprehensive error handling
- [ ] Implement retry logic for agent calls
- [ ] Add metrics and monitoring
