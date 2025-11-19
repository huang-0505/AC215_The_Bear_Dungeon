"""
app.py - Enhanced Orchestrator with Rule Validation and State Management

Orchestrates the D&D game flow with:
- Rule validation via Rule Agent
- State tree management
- Agent routing (Narrator, Combat)
- State transition detection
"""

import os
import logging
from uuid import uuid4
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import requests
from google import genai
from google.genai import types

from game_state import GameStateTree, GameStateType, AgentType
from rule_validator import RuleValidator
from context_builder import GameContextBuilder
from campaign_loader import CampaignLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Vertex AI for narrator agent
GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
NARRATOR_ENDPOINT_ID = os.getenv("NARRATOR_ENDPOINT_ID", "5165249441082376192")
NARRATOR_ENDPOINT = f"projects/{GCP_PROJECT}/locations/{GCP_LOCATION}/endpoints/{NARRATOR_ENDPOINT_ID}"

# Initialize GenAI client for narrator
llm_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

# Configuration for narrator generation
narrator_generation_config = types.GenerateContentConfig(
    max_output_tokens=8192,  # Maximum allowed for Gemini models
    temperature=0.8,
    top_p=0.95,
)

# Initialize FastAPI
app = FastAPI(
    title="D&D Game Orchestrator",
    description="Orchestrates game flow with rule validation and state management",
    version="2.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
RULE_AGENT_URL = os.getenv("RULE_AGENT_URL", "http://localhost:9002")
NARRATOR_AGENT_URL = os.getenv("NARRATOR_AGENT_URL", "http://localhost:9001")
COMBAT_AGENT_URL = os.getenv("COMBAT_AGENT_URL", "http://localhost:9000")

# Initialize services
rule_validator = RuleValidator(rule_agent_url=RULE_AGENT_URL)
context_builder = GameContextBuilder()

# In-memory session storage (use Redis in production)
game_sessions: Dict[str, GameStateTree] = {}


# ========== Pydantic Models ==========
class UserInput(BaseModel):
    text: str
    session_id: Optional[str] = None


class GameStartRequest(BaseModel):
    campaign_id: Optional[str] = None  # e.g., "stormwreck-isle"
    character_class: Optional[str] = None  # e.g., "Fighter"
    character_name: Optional[str] = None  # e.g., "Aragorn"
    initial_prompt: Optional[str] = None  # Custom prompt (overrides campaign)


class GameStateResponse(BaseModel):
    session_id: str
    state_type: str
    agent_used: str
    response: str
    validation: Optional[Dict] = None
    state_node: Dict
    transition: Optional[str] = None


# ========== Helper Functions ==========
def detect_combat_trigger(text: str) -> bool:
    """Use LLM to detect if narrative indicates combat start"""
    prompt = f"""
    Analyze this D&D narrative text and determine if it describes the START of a combat encounter.
    Look for phrases like:
    - "enemies appear", "ambush", "attack", "roll for initiative"
    - Monster/enemy descriptions appearing
    - Hostile NPCs engaging
    - "You are attacked"

    Text: "{text}"

    Answer with only "YES" or "NO".
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return "yes" in response.choices[0].message.content.lower()
    except Exception as e:
        logger.error(f"Error detecting combat trigger: {str(e)}")
        return False


def detect_combat_end(combat_state: Dict) -> bool:
    """Check if combat has ended based on combat state"""
    return combat_state.get("battle_over", False)


# ========== Agent Communication ==========
def call_narrator_agent(user_input: str, rules_context: Optional[str] = None) -> Dict:
    """Call the narrator agent (finetuned Gemini model on Vertex AI)"""
    try:
        logger.info(f"Calling narrator agent with input: {user_input}")

        # Build the prompt for the finetuned narrator
        prompt = f"Player action: {user_input}\n\nNarrate the outcome:"

        if rules_context:
            prompt = f"Player action: {user_input}\n\nRelevant D&D rules:\n{rules_context}\n\nNarrate the outcome:"

        # Call the finetuned model endpoint using genai client
        response = llm_client.models.generate_content(
            model=NARRATOR_ENDPOINT,
            contents=prompt,
            config=narrator_generation_config,
        )

        # Extract the narrative from response
        result = response.text if response.text else "The mists of magic obscure the tale..."

        return {"agent": "narrator", "result": result}

    except Exception as e:
        logger.error(f"Error calling narrator agent: {str(e)}")
        return {"agent": "narrator", "result": f"Narrator error: {str(e)}"}


def call_combat_agent_start(rules_context: Optional[str] = None) -> Dict:
    """Start a new combat session"""
    try:
        response = requests.post(
            f"{COMBAT_AGENT_URL}/combat/start",
            json={},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error starting combat: {str(e)}")
        return {
            "session_id": str(uuid4()),
            "message": "⚔️ Combat begins! (Combat agent unavailable)",
            "state": {"battle_over": False}
        }


def call_combat_agent_action(session_id: str, action: str, rules_context: Optional[str] = None) -> Dict:
    """Submit an action to the combat agent"""
    try:
        response = requests.post(
            f"{COMBAT_AGENT_URL}/combat/action/{session_id}",
            json={"action": action},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error in combat action: {str(e)}")
        return {
            "narrative": f"Combat action error: {str(e)}",
            "raw_result": "",
            "state": {"battle_over": True, "winner": "unknown"}
        }


# ========== State Handlers ==========
def handle_narration_action(tree: GameStateTree, current_node, data: UserInput, validation: Dict) -> Dict:
    """Handle action during narration state"""

    # Call narrator agent with rules context
    response = call_narrator_agent(
        data.text,
        rules_context=validation.get("rule_text")
    )
    current_node.agent_response = response["result"]
    current_node.narrative_text = response["result"]

    # Check for combat trigger
    if detect_combat_trigger(response["result"]):
        logger.info("Combat triggered during narration")

        combat_node = tree.add_child(
            parent_id=current_node.id,
            state_type=GameStateType.COMBAT,
            agent=AgentType.COMBAT,
            metadata={
                "trigger": data.text,
                "trigger_rules": validation.get("rule_text")
            }
        )

        # Start combat with rules context
        combat_start = call_combat_agent_start(
            rules_context=validation.get("rule_text")
        )
        combat_node.combat_session_id = combat_start["session_id"]
        combat_node.agent_response = combat_start.get("message", "Combat initiated!")

        tree.transition_to(combat_node.id)

        return {
            "session_id": data.session_id,
            "state_type": "combat",
            "agent_used": "combat",
            "response": combat_start.get("message", "Combat begins!"),
            "validation": validation,
            "state_node": combat_node.to_dict(),
            "transition": "narration -> combat"
        }

    return {
        "session_id": data.session_id,
        "state_type": "narration",
        "agent_used": "narrator",
        "response": response["result"],
        "validation": validation,
        "state_node": current_node.to_dict()
    }


def handle_combat_action(tree: GameStateTree, current_node, data: UserInput, validation: Dict) -> Dict:
    """Handle action during combat state"""

    # Call combat agent with rules context
    combat_response = call_combat_agent_action(
        session_id=current_node.combat_session_id,
        action=data.text,
        rules_context=validation.get("rule_text")
    )
    current_node.agent_response = combat_response.get("narrative", "Combat continues...")

    # Check for combat end
    if detect_combat_end(combat_response.get("state", {})):
        logger.info("Combat ended")

        narration_node = tree.add_child(
            parent_id=current_node.id,
            state_type=GameStateType.NARRATION,
            agent=AgentType.NARRATOR,
            metadata={
                "combat_outcome": combat_response["state"].get("winner"),
                "previous_combat_id": current_node.combat_session_id
            }
        )

        # Generate post-combat narration
        post_combat_prompt = f"Combat ended. Winner: {combat_response['state'].get('winner')}. Continue the story."
        narrator_response = call_narrator_agent(post_combat_prompt)
        narration_node.agent_response = narrator_response["result"]

        tree.transition_to(narration_node.id)

        return {
            "session_id": data.session_id,
            "state_type": "narration",
            "agent_used": "narrator",
            "response": narrator_response["result"],
            "validation": validation,
            "state_node": narration_node.to_dict(),
            "transition": "combat -> narration",
            "combat_summary": combat_response
        }

    return {
        "session_id": data.session_id,
        "state_type": "combat",
        "agent_used": "combat",
        "response": combat_response.get("narrative", ""),
        "validation": validation,
        "state_node": current_node.to_dict(),
        "combat_state": combat_response.get("state", {})
    }


# ========== API Routes ==========
@app.get("/")
async def root():
    return {
        "service": "D&D Game Orchestrator",
        "version": "2.0",
        "features": ["rule_validation", "state_management", "agent_routing"]
    }


@app.get("/health")
async def health_check():
    """Health check with service status"""
    return {
        "status": "healthy",
        "services": {
            "rule_agent": rule_validator.check_health(),
            "active_sessions": len(game_sessions)
        }
    }


@app.get("/campaigns")
def list_campaigns():
    """Get list of available campaigns"""
    return {
        "campaigns": CampaignLoader.list_campaigns()
    }


@app.get("/campaigns/{campaign_id}")
def get_campaign_details(campaign_id: str):
    """Get detailed information about a specific campaign"""
    campaign = CampaignLoader.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")
    return campaign.to_dict()


@app.post("/game/start")
def start_game(request: GameStartRequest):
    """
    Initialize a new game session with optional campaign.

    Supports:
    - Pre-loaded campaigns (e.g., "stormwreck-isle")
    - Custom character creation
    - Custom starting prompts

    Examples:
    1. Start Dragons of Stormwreck Isle:
       {"campaign_id": "stormwreck-isle", "character_class": "Fighter", "character_name": "Thorin"}

    2. Custom adventure:
       {"initial_prompt": "You wake up in a dark dungeon..."}
    """
    session_id = str(uuid4())
    tree = GameStateTree()
    root = tree.create_root(GameStateType.NARRATION)

    # Determine initial prompt
    if request.campaign_id:
        # Load pre-defined campaign
        try:
            campaign_data = CampaignLoader.initialize_campaign(
                request.campaign_id,
                request.character_class,
                request.character_name
            )
            initial_prompt = campaign_data["initial_prompt"]

            # Store campaign metadata in root node
            root.metadata.update({
                "campaign_id": campaign_data["campaign_id"],
                "campaign_name": campaign_data["campaign_name"],
                "starting_location": campaign_data["starting_location"],
                "initial_quest": campaign_data["initial_quest"],
                **campaign_data["metadata"]
            })

            logger.info(f"Starting campaign: {campaign_data['campaign_name']}")

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    elif request.initial_prompt:
        # Custom prompt provided
        initial_prompt = request.initial_prompt
        root.metadata.update({
            "campaign_type": "custom",
            "character_class": request.character_class,
            "character_name": request.character_name
        })

    else:
        # Default tavern start
        initial_prompt = "Start a new D&D adventure in a fantasy tavern."
        root.metadata["campaign_type"] = "default"

    # Use the pre-written campaign opening directly (no need to call narrator)
    # The narrator will be used for subsequent player actions
    root.narrative_text = initial_prompt
    root.agent_response = initial_prompt
    root.player_action = None  # No player action yet, this is the campaign intro

    # Check if combat immediately triggered
    if detect_combat_trigger(initial_prompt):
        root.transition_triggered = True
        root.next_state_type = GameStateType.COMBAT

    game_sessions[session_id] = tree

    logger.info(f"Started new game session: {session_id}")

    return {
        "session_id": session_id,
        "state": root.to_dict(),
        "response": initial_prompt,
        "campaign_info": root.metadata,
        "message": "Game started successfully!"
    }


@app.post("/game/action")
def game_action(data: UserInput):
    """
    Handle player action with full validation pipeline.

    Flow:
    1. Validate action with Rule Agent
    2. Check for sabotage/invalid actions
    3. Route to appropriate agent (Narrator/Combat)
    4. Detect state transitions
    5. Update game tree
    """

    if not data.session_id or data.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new game first.")

    tree = game_sessions[data.session_id]
    current_node = tree.get_current()

    if not current_node:
        raise HTTPException(status_code=500, detail="Invalid game state")

    logger.info(f"Session {data.session_id}: Processing action '{data.text}' in state {current_node.state_type.value}")

    # ========== STEP 1: RULE VALIDATION ==========
    game_context = context_builder.build_context(tree)
    validation = rule_validator.validate_action(data.text, game_context)

    # Store validation in node
    current_node.rule_validation = validation
    current_node.was_validated = True

    logger.info(f"Validation result: {validation.get('validation_type')}")

    # ========== STEP 2: HANDLE SABOTAGE ==========
    if rule_validator.is_sabotage(validation):
        logger.warning(f"Sabotage detected: {data.text}")
        return {
            "session_id": data.session_id,
            "error": "invalid_action",
            "validation": validation,
            "message": (
                f"Your input: '{data.text}'\n\n"
                "This appears to be a meta-game or sabotage attempt. "
                "Please provide an in-character action that follows D&D rules."
            )
        }

    # ========== STEP 3: HANDLE INVALID ACTIONS (if needed) ==========
    # Note: Current Rule Agent informs but doesn't reject
    # Uncomment below if you want to block invalid actions
    # if not validation.get("is_valid", True):
    #     current_node.validation_errors.append(validation.get("explanation"))
    #     return {
    #         "session_id": data.session_id,
    #         "error": "rule_violation",
    #         "validation": validation,
    #         "message": f"Action not allowed: {validation.get('explanation')}"
    #     }

    # ========== STEP 4: ACTION IS VALID - ROUTE TO AGENT ==========
    current_node.player_action = data.text
    current_node.applicable_rules = validation.get("rule_text")

    # Route based on current state
    if current_node.state_type == GameStateType.NARRATION:
        return handle_narration_action(tree, current_node, data, validation)

    elif current_node.state_type == GameStateType.COMBAT:
        return handle_combat_action(tree, current_node, data, validation)

    return {"error": "Unknown state type"}


@app.get("/game/state/{session_id}")
def get_game_state(session_id: str):
    """
    Get current game state and full history.

    Returns:
    - Current state node
    - Path from root to current
    - Full game tree
    """
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    tree = game_sessions[session_id]

    return {
        "session_id": session_id,
        "current_state": tree.get_current().to_dict() if tree.get_current() else None,
        "path": [node.to_dict() for node in tree.get_path_from_root()],
        "story_summary": context_builder.get_story_summary(tree),
        "full_tree": tree.to_dict()
    }


@app.delete("/game/session/{session_id}")
def end_game_session(session_id: str):
    """End a game session and clean up"""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del game_sessions[session_id]
    logger.info(f"Ended game session: {session_id}")

    return {"message": "Game session ended", "session_id": session_id}


# ========== Legacy Endpoints (for backward compatibility) ==========
@app.post("/agent/narration")
def narrator_agent_legacy(data: UserInput):
    """Legacy narrator endpoint"""
    return call_narrator_agent(data.text)


@app.post("/orchestrate")
def orchestrate_legacy(data: UserInput):
    """
    Legacy orchestrate endpoint.

    Note: This is kept for backward compatibility.
    New clients should use /game/start and /game/action instead.
    """
    logger.warning("Using legacy /orchestrate endpoint. Consider migrating to /game/action")

    # Simple intent classification
    prompt = f"""
    Classify the following D&D player input into one of two categories:
    - narration
    - combat

    Input: "{data.text}"
    Output:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    intent = response.choices[0].message.content.strip().lower()

    if "combat" in intent:
        result = {"agent": "combat", "result": f"⚔️ Combat agent received: {data.text}"}
    else:
        result = call_narrator_agent(data.text)

    return {
        "orchestrator_intent": intent,
        "agent_response": result
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
