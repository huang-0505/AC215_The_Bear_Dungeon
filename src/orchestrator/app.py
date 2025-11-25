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
from typing import Dict, Optional, List

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
from story_tree_loader import StoryTreeLoader, StoryTree

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. OpenAI features may not work.")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize Vertex AI for narrator agent
GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
NARRATOR_ENDPOINT_ID = os.getenv("NARRATOR_ENDPOINT_ID", "5165249441082376192")

if not GCP_PROJECT:
    logger.warning("GCP_PROJECT not set. Vertex AI narrator features may not work.")
    NARRATOR_ENDPOINT = None
    llm_client = None
else:
    NARRATOR_ENDPOINT = f"projects/{GCP_PROJECT}/locations/{GCP_LOCATION}/endpoints/{NARRATOR_ENDPOINT_ID}"
    # Initialize GenAI client for narrator
    try:
        llm_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    except Exception as e:
        logger.error(f"Failed to initialize GenAI client: {str(e)}")
        llm_client = None

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

# Story tree storage (maps session_id to StoryTree)
story_trees: Dict[str, StoryTree] = {}

# Current story node tracking (maps session_id to current node_id in story tree)
current_story_nodes: Dict[str, str] = {}


# ========== Pydantic Models ==========
class UserInput(BaseModel):
    text: str
    session_id: Optional[str] = None


class CombatActionRequest(BaseModel):
    action: str


class GameStartRequest(BaseModel):
    campaign_id: Optional[str] = None  # e.g., "stormwreck-isle"
    character_class: Optional[str] = None  # e.g., "Fighter"
    character_name: Optional[str] = None  # e.g., "Aragorn"
    initial_prompt: Optional[str] = None  # Custom prompt (overrides campaign)
    max_combats: Optional[int] = 5  # Game ends after this many combats (default: 5)
    combat_rounds: Optional[List[int]] = None  # Rounds where combat is forced (default: [3, 10, 15])


class GameStateResponse(BaseModel):
    session_id: str
    state_type: str
    agent_used: str
    response: str
    validation: Optional[Dict] = None
    state_node: Dict
    transition: Optional[str] = None
    choices: Optional[List[str]] = None  # Tree-structure mode: suggested choices


# ========== Helper Functions ==========
def detect_combat_trigger(text: str) -> bool:
    """Use LLM to detect if narrative indicates combat start"""
    if not client:
        logger.warning("OpenAI client not available, cannot detect combat trigger")
        return False
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
def call_narrator_agent(user_input: str, rules_context: Optional[str] = None, generate_choices: bool = True) -> Dict:
    """Call the narrator agent (finetuned Gemini model on Vertex AI)"""
    try:
        if not llm_client or not NARRATOR_ENDPOINT:
            logger.warning("Narrator agent not available (missing GCP configuration), using fallback")
            return {
                "agent": "narrator",
                "result": f"You attempt: {user_input}. The story continues, but the narrator is temporarily unavailable.",
                "choices": None
            }
        
        logger.info(f"Calling narrator agent with input: {user_input}")

        # Build the prompt for the finetuned narrator
        prompt = f"Player action: {user_input}\n\nNarrate the outcome:"
        
        if generate_choices:
            prompt += "\n\nAfter your narration, provide 3-4 suggested action choices for the player. Format choices as:\nCHOICES:\n1. [choice 1]\n2. [choice 2]\n3. [choice 3]"

        if rules_context:
            prompt = f"Player action: {user_input}\n\nRelevant D&D rules:\n{rules_context}\n\nNarrate the outcome:"
            if generate_choices:
                prompt += "\n\nAfter your narration, provide 3-4 suggested action choices for the player. Format choices as:\nCHOICES:\n1. [choice 1]\n2. [choice 2]\n3. [choice 3]"

        # Call the finetuned model endpoint using genai client
        response = llm_client.models.generate_content(
            model=NARRATOR_ENDPOINT,
            contents=prompt,
            config=narrator_generation_config,
        )

        # Extract the narrative from response
        result = response.text if response.text else "The mists of magic obscure the tale..."

        # Extract choices if present
        choices = extract_choices_from_text(result)
        if choices:
            # Remove choices section from narrative
            result = remove_choices_from_text(result)

        return {"agent": "narrator", "result": result, "choices": choices}

    except Exception as e:
        logger.error(f"Error calling narrator agent: {str(e)}")
        return {"agent": "narrator", "result": f"Narrator error: {str(e)}", "choices": None}


def extract_choices_from_text(text: str) -> Optional[List[str]]:
    """Extract choice options from narrator response"""
    import re
    
    # Look for CHOICES: section
    choices_match = re.search(r'CHOICES:\s*\n((?:\d+\.\s*[^\n]+\n?)+)', text, re.IGNORECASE | re.MULTILINE)
    if choices_match:
        choices_text = choices_match.group(1)
        # Extract individual choices
        choices = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n*$)', choices_text, re.MULTILINE)
        choices = [choice.strip() for choice in choices if choice.strip()]
        if choices:
            return choices
    
    # Fallback: Use LLM to extract choices if format not found
    try:
        if not client:
            logger.warning("OpenAI client not available, cannot extract choices")
            return None
        prompt = f"""
        Extract 3-4 suggested action choices from this D&D narrative text.
        Return a JSON object with a "choices" key containing an array of strings.
        
        Text: {text}
        
        Example format: {{"choices": ["Investigate the door", "Search for traps", "Try to pick the lock"]}}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        import json
        result = json.loads(response.choices[0].message.content)
        # Handle both {"choices": [...]} and direct array formats
        if isinstance(result, list):
            return result[:4]  # Limit to 4 choices
        elif "choices" in result and isinstance(result["choices"], list):
            return result["choices"][:4]  # Limit to 4 choices
    except Exception as e:
        logger.warning(f"Failed to extract choices with LLM: {str(e)}")
    
    return None


def remove_choices_from_text(text: str) -> str:
    """Remove the CHOICES section from narrative text"""
    import re
    # Remove CHOICES: section
    text = re.sub(r'CHOICES:\s*\n((?:\d+\.\s*[^\n]+\n?)+)', '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


def get_player_stats_by_class(character_class: Optional[str]) -> Dict:
    """Get player stats based on character class choice"""
    class_stats = {
        "Fighter": {"name": "Fighter", "hp": 20, "ac": 18, "attributes": {"STR": 4, "DEX": 2, "INT": 1}, "attack_bonus": 7, "damage": 15},
        "Wizard": {"name": "Wizard", "hp": 14, "ac": 12, "attributes": {"STR": 1, "DEX": 2, "INT": 5}, "attack_bonus": 3, "damage": 12},
        "Ranger": {"name": "Ranger", "hp": 16, "ac": 15, "attributes": {"STR": 3, "DEX": 4, "INT": 2}, "attack_bonus": 6, "damage": 10},
        "Cleric": {"name": "Cleric", "hp": 15, "ac": 16, "attributes": {"STR": 2, "DEX": 2, "INT": 4}, "attack_bonus": 4, "damage": 8},
        "Barbarian": {"name": "Barbarian", "hp": 26, "ac": 14, "attributes": {"STR": 5, "DEX": 3, "INT": 1}, "attack_bonus": 8, "damage": 18},
        "Rogue": {"name": "Rogue", "hp": 14, "ac": 15, "attributes": {"STR": 2, "DEX": 5, "INT": 3}, "attack_bonus": 5, "damage": 9},
    }
    return class_stats.get(character_class or "Fighter", class_stats["Fighter"])

def call_combat_agent_start(rules_context: Optional[str] = None, character_class: Optional[str] = None) -> Dict:
    """Start a new combat session with character-based player stats and fixed enemies"""
    try:
        # Get player stats based on character class
        player_stats = get_player_stats_by_class(character_class)
        
        # Fixed enemies for all combats (same stats every time)
        enemies = [
            {"name": "Goblin", "hp": 12, "ac": 13, "attributes": {"DEX": 3}, "attack_bonus": 3, "damage": 6, "role": "enemy"},
            {"name": "Troll", "hp": 16, "ac": 13, "attributes": {"STR": 4, "DEX": 2}, "attack_bonus": 5, "damage": 8, "role": "enemy"},
            {"name": "Dragon", "hp": 20, "ac": 20, "attributes": {"STR": 6, "DEX": 6, "INT": 6}, "attack_bonus": 8, "damage": 12, "role": "enemy"}
        ]
        
        # Create request with player and fixed enemies
        request_data = {
            "players": [player_stats],
            "enemies": enemies
        }
        
        logger.info(f"Starting combat with player: {player_stats['name']} (HP: {player_stats['hp']}, AC: {player_stats['ac']})")
        
        response = requests.post(
            f"{COMBAT_AGENT_URL}/combat/start",
            json=request_data,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Combat started successfully with session_id: {result.get('session_id')}")
        return result
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
            timeout=15  # Increased to 15 seconds to account for GenAI processing (3s timeout + buffer)
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # If battle is already over, check state instead
        if e.response and e.response.status_code == 400:
            logger.info(f"Combat action failed (likely battle over), checking state...")
            try:
                state_response = requests.get(
                    f"{COMBAT_AGENT_URL}/combat/state/{session_id}",
                    timeout=15  # Increased timeout for consistency
                )
                state_response.raise_for_status()
                state_data = state_response.json()
                if state_data.get("battle_over"):
                    return {
                        "narrative": "The battle has ended!",
                        "raw_result": "Battle over",
                        "state": state_data
                    }
            except:
                pass
        logger.error(f"Error in combat action: {str(e)}")
        return {
            "narrative": f"Combat action error: {str(e)}",
            "raw_result": "",
            "state": {"battle_over": True, "winner": "unknown"}
        }
    except Exception as e:
        logger.error(f"Error in combat action: {str(e)}")
        return {
            "narrative": f"Combat action error: {str(e)}",
            "raw_result": "",
            "state": {"battle_over": True, "winner": "unknown"}
        }


# ========== State Handlers ==========
def handle_narration_action(tree: GameStateTree, current_node, data: UserInput, validation: Dict, session_id: str) -> Dict:
    """Handle action during narration state"""

    # Increment narration round
    tree.increment_narration_round()
    
    # Check for round-based combat trigger FIRST (before processing action)
    if tree.should_trigger_combat():
        logger.info(f"Combat forced at round {tree.narration_round}")

        combat_node = tree.add_child(
            parent_id=current_node.id,
            state_type=GameStateType.COMBAT,
            agent=AgentType.COMBAT,
            metadata={
                "trigger": "round_based",
                "narration_round": tree.narration_round,
                "trigger_rules": validation.get("rule_text")
            }
        )

        # Get character class from game tree metadata
        root_node = tree.get_node(tree.root_id) if tree.root_id else None
        character_class = root_node.metadata.get("character_class") if root_node else None
        
        # Start combat with rules context and character class
        combat_start = call_combat_agent_start(
            rules_context=validation.get("rule_text"),
            character_class=character_class
        )
        combat_node.combat_session_id = combat_start["session_id"]
        combat_node.agent_response = combat_start.get("message", "⚔️ Combat begins!")

        tree.transition_to(combat_node.id)

        return {
            "session_id": session_id,
            "state_type": "combat",
            "agent_used": "combat",
            "response": f"⚔️ **Combat Encounter at Round {tree.narration_round}!**\n\n{combat_start.get('message', 'Combat begins! Roll for initiative!')}",
            "validation": validation,
            "state_node": combat_node.to_dict(),
            "transition": "narration -> combat",
            "choices": None,
            "narration_round": tree.narration_round,
            "combat_count": tree.combat_count,
            "combat_session_id": combat_start["session_id"]  # Include combat session ID for frontend redirect
        }
    
    # Check if game should end (max combats reached)
    if tree.should_end_game():
        ending_text = f"🎉 **ADVENTURE COMPLETE!** 🎉\n\nYou have completed {tree.combat_count} combat encounters! Your journey has come to an end. Well done, adventurer!\n\n*The adventure ends here. Thank you for playing!*"
        current_node.state_type = GameStateType.GAME_OVER
        current_node.agent_response = ending_text
        current_node.metadata["is_ending"] = True
        current_node.metadata["ending_type"] = "victory"
        
        return {
            "session_id": session_id,
            "state_type": "game_over",
            "agent_used": "narrator",
            "response": ending_text,
            "validation": validation,
            "state_node": current_node.to_dict(),
            "choices": [],
            "is_ending": True,
            "ending_type": "victory",
            "combat_available": False
        }
    
    # Check for combat option (always available)
    if data.text.lower().strip() in ["⚔️ enter combat", "enter combat", "combat", "fight", "⚔️ combat", "⚔️ enter combat (not available)"]:
        # Check if combat is actually available
        story_tree = story_trees.get(session_id)
        current_story_node_id = current_story_nodes.get(session_id)
        combat_available = False
        
        # Check round-based combat trigger
        if tree.should_trigger_combat():
            combat_available = True
            logger.info(f"Combat triggered at round {tree.narration_round}")
        
        # Check story tree combat availability
        if not combat_available and story_tree and current_story_node_id:
            story_node = story_tree.get_node(current_story_node_id)
            if story_node:
                combat_available = story_node.combat_available
        
        # Check metadata
        if not combat_available:
            combat_available = current_node.metadata.get("combat_available", False)
        
        if not combat_available:
            # Combat not available yet
            return {
                "session_id": session_id,
                "state_type": "narration",
                "agent_used": "narrator",
                "response": "⚔️ **No combat available yet.**\n\nThe current situation doesn't present any immediate threats or combat opportunities. Continue exploring to find enemies!",
                "validation": validation,
                "state_node": current_node.to_dict(),
                "choices": _get_choices_with_combat(current_node, story_tree, current_story_node_id),
                "combat_available": False
            }
        else:
            # Combat is available - trigger combat
            logger.info(f"Combat triggered at round {tree.narration_round}")
            
            combat_node = tree.add_child(
                parent_id=current_node.id,
                state_type=GameStateType.COMBAT,
                agent=AgentType.COMBAT,
                metadata={
                    "trigger": "player_requested",
                    "story_node_id": current_story_node_id,
                    "narration_round": tree.narration_round
                }
            )
            
            # Get character class from game tree metadata
            root_node = tree.get_node(tree.root_id) if tree.root_id else None
            character_class = root_node.metadata.get("character_class") if root_node else None
            
            # Start combat with rules context and character class
            combat_start = call_combat_agent_start(
                rules_context=validation.get("rule_text"),
                character_class=character_class
            )
            combat_node.combat_session_id = combat_start["session_id"]
            combat_node.agent_response = combat_start.get("message", "⚔️ Combat begins!")
            
            tree.transition_to(combat_node.id)
            
            return {
                "session_id": session_id,
                "state_type": "combat",
                "agent_used": "combat",
                "response": combat_start.get("message", "⚔️ Combat begins! Roll for initiative!"),
                "validation": validation,
                "state_node": combat_node.to_dict(),
                "transition": "narration -> combat",
                "combat_session_id": combat_start["session_id"],
                "combat_state": combat_start.get("state", {}),
                "choices": None,
                "narration_round": tree.narration_round,
                "combat_count": tree.combat_count
            }
    
    # Check if we have a story tree for this session
    story_tree = story_trees.get(session_id)
    current_story_node_id = current_story_nodes.get(session_id)
    
    response_text = None
    response_choices = None
    is_ending = False
    ending_type = None
    next_story_node_id = None
    combat_available = False
    
    if story_tree and current_story_node_id:
        # Try to find next node based on player's choice
        story_node = story_tree.get_node(current_story_node_id)
        if story_node:
            # Check if this is an ending node
            if story_node.is_ending:
                is_ending = True
                ending_type = story_node.ending_type
                response_text = story_node.narrative
                response_choices = []  # No choices at endings
                current_node.metadata["is_ending"] = True
                current_node.metadata["ending_type"] = ending_type
                current_node.state_type = GameStateType.GAME_OVER
            else:
                # Find next node based on choice
                next_story_node = story_tree.get_next_node_for_choice(current_story_node_id, data.text)
                
                if next_story_node:
                    # Use predefined narrative
                    response_text = next_story_node.narrative
                    response_choices = next_story_node.choices
                    next_story_node_id = next_story_node.node_id
                    is_ending = next_story_node.is_ending
                    ending_type = next_story_node.ending_type
                    combat_available = next_story_node.combat_available
                    
                    # Update current story node
                    current_story_nodes[session_id] = next_story_node_id
                    current_node.metadata["story_node_id"] = next_story_node_id
                    current_node.metadata["is_ending"] = is_ending
                    current_node.metadata["combat_available"] = combat_available
                    if is_ending:
                        current_node.metadata["ending_type"] = ending_type
                        current_node.state_type = GameStateType.GAME_OVER
                else:
                    # No matching node found, use AI narrator but guide toward story tree
                    logger.info(f"No matching story node for choice: {data.text}, using AI narrator")
                    response = call_narrator_agent(
                        data.text,
                        rules_context=validation.get("rule_text"),
                        generate_choices=True
                    )
                    response_text = response["result"]
                    response_choices = response.get("choices")
                    
                    # Try to find a nearby node in the story tree
                    keywords = data.text.split()[:5]  # Use first few words as keywords
                    nearby_node = story_tree.find_node_by_keywords(keywords, current_story_node_id)
                    if nearby_node:
                        logger.info(f"Found nearby story node: {nearby_node.node_id}, guiding narrative")
                        # Blend AI response with story guidance
                        response_text = f"{response_text}\n\n{nearby_node.narrative[:200]}..."
                        response_choices = nearby_node.choices if nearby_node.choices else response_choices
                        combat_available = nearby_node.combat_available
    
    if not response_text:
        # No story tree or no match, use AI narrator
        response = call_narrator_agent(
            data.text,
            rules_context=validation.get("rule_text"),
            generate_choices=True
        )
        response_text = response["result"]
        response_choices = response.get("choices")
    
    current_node.agent_response = response_text
    current_node.narrative_text = response_text
    
    # Store choices in metadata for tree-structure mode
    if response_choices:
        current_node.metadata["choices"] = response_choices

    # Round-based combat trigger is now checked at the start of handle_narration_action
    # (moved up to line 340 to trigger immediately after round increment)

    # Check for AI-detected combat trigger (fallback)
    if detect_combat_trigger(response_text):
        logger.info("Combat triggered by AI detection during narration")

        combat_node = tree.add_child(
            parent_id=current_node.id,
            state_type=GameStateType.COMBAT,
            agent=AgentType.COMBAT,
            metadata={
                "trigger": "ai_detected",
                "narration_round": tree.narration_round,
                "trigger_rules": validation.get("rule_text")
            }
        )

        # Get character class from game tree metadata
        root_node = tree.get_node(tree.root_id) if tree.root_id else None
        character_class = root_node.metadata.get("character_class") if root_node else None
        
        # Start combat with rules context and character class
        combat_start = call_combat_agent_start(
            rules_context=validation.get("rule_text"),
            character_class=character_class
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
            "transition": "narration -> combat",
            "choices": None,
            "narration_round": tree.narration_round,
            "combat_count": tree.combat_count
        }

    # Determine combat availability (check round-based triggers)
    if tree.should_trigger_combat():
        combat_available = True
        logger.info(f"Combat available at round {tree.narration_round}")
    
    # Always add combat option to choices
    final_choices = _get_choices_with_combat(current_node, story_tree, current_story_node_id, response_choices, tree)
    
    # Ensure we always have choices (generate default if none)
    if not final_choices or len(final_choices) == 0:
        # Generate default choices if AI didn't provide any
        final_choices = [
            "Continue exploring",
            "Investigate the area",
            "Search for clues",
            "⚔️ Enter Combat (Not Available)" if not combat_available else "⚔️ Enter Combat"
        ]

    return {
        "session_id": data.session_id,
        "state_type": "game_over" if is_ending else "narration",
        "agent_used": "narrator",
        "response": response_text,
        "validation": validation,
        "state_node": current_node.to_dict(),
        "choices": final_choices,  # Include choices with combat option
        "is_ending": is_ending,
        "ending_type": ending_type,
        "combat_available": combat_available,
        "narration_round": tree.narration_round,
        "combat_count": tree.combat_count,
        "max_combats": tree.max_combats
    }


def _get_choices_with_combat(current_node, story_tree, current_story_node_id: Optional[str], base_choices: Optional[List[str]] = None, game_tree: Optional[GameStateTree] = None) -> List[str]:
    """Add combat option to choices list"""
    choices = base_choices if base_choices else []
    
    # Get combat availability from multiple sources
    combat_available = False
    
    # Check round-based combat triggers
    if game_tree and game_tree.should_trigger_combat():
        combat_available = True
        logger.info(f"Combat available at round {game_tree.narration_round}")
    
    # Check story tree combat availability
    if not combat_available and story_tree and current_story_node_id:
        story_node = story_tree.get_node(current_story_node_id)
        if story_node:
            combat_available = story_node.combat_available
    
    # Check metadata for combat availability
    if not combat_available:
        combat_available = current_node.metadata.get("combat_available", False)
    
    # Add combat option (always show, but indicate availability)
    if combat_available:
        choices.append("⚔️ Enter Combat")
    else:
        choices.append("⚔️ Enter Combat (Not Available)")
    
    return choices


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

        # Increment combat count
        tree.increment_combat_count()
        
        # Check if game should end after this combat
        if tree.should_end_game():
            ending_text = f"🎉 **ADVENTURE COMPLETE!** 🎉\n\nYou have completed {tree.combat_count} combat encounters! Your journey has come to an end. Well done, adventurer!\n\n*The adventure ends here. Thank you for playing!*"
            narration_node.state_type = GameStateType.GAME_OVER
            narration_node.agent_response = ending_text
            narration_node.metadata["is_ending"] = True
            narration_node.metadata["ending_type"] = "victory"
            
            tree.transition_to(narration_node.id)
            
            return {
                "session_id": data.session_id,
                "state_type": "game_over",
                "agent_used": "narrator",
                "response": ending_text,
                "validation": validation,
                "state_node": narration_node.to_dict(),
                "transition": "combat -> narration -> game_over",
                "combat_summary": combat_response,
                "choices": [],
                "is_ending": True,
                "ending_type": "victory",
                "combat_count": tree.combat_count,
                "max_combats": tree.max_combats
            }

        # Generate post-combat narration
        post_combat_prompt = f"Combat ended. Winner: {combat_response['state'].get('winner')}. Continue the story."
        narrator_response = call_narrator_agent(post_combat_prompt, generate_choices=True)
        narration_node.agent_response = narrator_response["result"]
        if narrator_response.get("choices"):
            narration_node.metadata["choices"] = narrator_response["choices"]

        tree.transition_to(narration_node.id)
        
        # Add combat option to choices
        story_tree = story_trees.get(data.session_id)
        current_story_node_id = current_story_nodes.get(data.session_id)
        final_choices = _get_choices_with_combat(narration_node, story_tree, current_story_node_id, narrator_response.get("choices"), tree)

        return {
            "session_id": data.session_id,
            "state_type": "narration",
            "agent_used": "narrator",
            "response": narrator_response["result"],
            "validation": validation,
            "state_node": narration_node.to_dict(),
            "transition": "combat -> narration",
            "combat_summary": combat_response,
            "choices": final_choices,
            "narration_round": tree.narration_round,
            "combat_count": tree.combat_count,
            "max_combats": tree.max_combats
        }

    return {
        "session_id": data.session_id,
        "state_type": "combat",
        "agent_used": "combat",
        "response": combat_response.get("narrative", ""),
        "validation": validation,
        "state_node": current_node.to_dict(),
        "combat_state": combat_response.get("state", {}),
        "choices": None  # Combat doesn't generate choices
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
    
    # Configure combat settings
    if request.max_combats:
        tree.max_combats = request.max_combats
    if request.combat_rounds:
        tree.combat_rounds = request.combat_rounds
    
    root = tree.create_root(GameStateType.NARRATION)
    
    # Store character info in root metadata for combat initialization
    root.metadata["character_class"] = request.character_class
    root.metadata["character_name"] = request.character_name

    # Determine initial prompt
    story_tree = None
    current_story_node_id = None
    
    if request.campaign_id:
        # Load pre-defined campaign
        try:
            campaign_data = CampaignLoader.initialize_campaign(
                request.campaign_id,
                request.character_class,
                request.character_name
            )
            
            # Try to load story tree for this campaign
            story_tree = StoryTreeLoader.load_story_tree(request.campaign_id)
            if story_tree:
                story_trees[session_id] = story_tree
                current_story_node_id = story_tree.root_node_id
                current_story_nodes[session_id] = current_story_node_id
                
                # Use predefined narrative from story tree if available
                story_node = story_tree.get_root()
                if story_node:
                    initial_prompt = story_node.narrative
                    root.metadata["story_node_id"] = story_node.node_id
                    root.metadata["story_choices"] = story_node.choices
                    root.metadata["is_ending"] = story_node.is_ending
                    root.metadata["combat_available"] = story_node.combat_available
                    if story_node.is_ending:
                        root.metadata["ending_type"] = story_node.ending_type
            else:
                initial_prompt = campaign_data["initial_prompt"]
                logger.info(f"No story tree found for {request.campaign_id}, using free-form mode")

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

    # Use the pre-written campaign opening, but generate choices for the first round
    root.narrative_text = initial_prompt
    root.agent_response = initial_prompt
    root.player_action = None  # No player action yet, this is the campaign intro

    # Generate choices for the initial prompt using narrator
    initial_choices = []
    try:
        narrator_response = call_narrator_agent(
            initial_prompt,
            generate_choices=True
        )
        
        # Extract choices from narrator response
        initial_choices = narrator_response.get("choices", [])
        if initial_choices:
            root.metadata["story_choices"] = initial_choices
            logger.info(f"Generated {len(initial_choices)} initial choices")
        else:
            logger.warning("No choices generated by narrator, using empty list")
    except Exception as e:
        logger.error(f"Error generating initial choices: {str(e)}")
        initial_choices = []  # Fallback to empty choices

    # Check if combat immediately triggered
    if detect_combat_trigger(initial_prompt):
        root.transition_triggered = True
        root.next_state_type = GameStateType.COMBAT

    game_sessions[session_id] = tree

    logger.info(f"Started new game session: {session_id}")

    # Add combat option to initial choices
    combat_available = root.metadata.get("combat_available", False)
    # Check round-based combat trigger
    if tree.should_trigger_combat():
        combat_available = True
    
    if combat_available:
        initial_choices.append("⚔️ Enter Combat")
    else:
        initial_choices.append("⚔️ Enter Combat (Not Available)")

    return {
        "session_id": session_id,
        "state": root.to_dict(),
        "response": initial_prompt,
        "campaign_info": root.metadata,
        "choices": initial_choices,  # Include choices with combat option
        "is_ending": root.metadata.get("is_ending", False),
        "combat_available": combat_available,
        "narration_round": tree.narration_round,
        "combat_count": tree.combat_count,
        "max_combats": tree.max_combats,
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
        return handle_narration_action(tree, current_node, data, validation, data.session_id)

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


@app.get("/combat/state/{combat_session_id}")
def get_combat_state(combat_session_id: str):
    """Proxy endpoint to get combat state from combat agent"""
    try:
        logger.info(f"Fetching combat state for combat session: {combat_session_id}")
        response = requests.get(
            f"{COMBAT_AGENT_URL}/combat/state/{combat_session_id}",
            timeout=15  # Consistent timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"Combat agent returned error: {str(e)}")
        if e.response:
            error_detail = e.response.json().get("detail", "Combat session not found") if e.response.content else "Combat session not found"
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        raise HTTPException(status_code=500, detail="Failed to get combat state")
    except Exception as e:
        logger.error(f"Error getting combat state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get combat state: {str(e)}")


@app.post("/combat/action/{combat_session_id}")
def submit_combat_action(combat_session_id: str, action_data: CombatActionRequest):
    """Proxy endpoint to submit combat action to combat agent"""
    try:
        logger.info(f"Submitting combat action for session {combat_session_id}: {action_data.action}")
        result = call_combat_agent_action(combat_session_id, action_data.action)
        return result
    except Exception as e:
        logger.error(f"Error submitting combat action: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit combat action: {str(e)}")


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
    if not client:
        logger.warning("OpenAI client not available, defaulting to narration")
        intent = "narration"
    else:
        prompt = f"""
        Classify the following D&D player input into one of two categories:
        - narration
        - combat

        Input: "{data.text}"
        Output:
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            intent = response.choices[0].message.content.strip().lower()
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}")
            intent = "narration"

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
