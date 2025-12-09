import os
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from api.utils.combat_engine import Character, CombatEngine
from api.utils.combat_ai import ActionParser, ActionParserBot, DnDBot, DnDNarrator

# Define Router
router = APIRouter()

# In-memory session storage (in production, use Redis or database)
combat_sessions: Dict[str, CombatEngine] = {}
action_parsers: Dict[str, ActionParser] = {}
bot_parsers: Dict[str, ActionParserBot] = {}
bots: Dict[str, DnDBot] = {}
narrators: Dict[str, DnDNarrator] = {}


# ========== Pydantic Models ==========
class CharacterData(BaseModel):
    name: str
    hp: int
    ac: int
    attributes: Dict[str, int]
    attack_bonus: int
    damage: int
    role: str = "player"


class InitiateCombatRequest(BaseModel):
    players: Optional[List[CharacterData]] = None
    enemies: Optional[List[CharacterData]] = None


class PlayerActionRequest(BaseModel):
    action: str  # Natural language action description


class CombatState(BaseModel):
    session_id: str
    round: int
    current_actor: Optional[str]
    players: List[Dict]
    enemies: List[Dict]
    battle_over: bool
    winner: Optional[str] = None


class ActionResponse(BaseModel):
    narrative: str
    raw_result: str
    state: CombatState


# ========== Helper Functions ==========
def create_default_players() -> List[Character]:
    """Create default player party."""
    return [
        Character("Knight", 0, 20, 24, {"STR": 4, "DEX": 2, "INT": 1}, attack_bonus=7, damage=15),
        Character("Wizard", 1, 14, 8, {"STR": 1, "DEX": 2, "INT": 5}, attack_bonus=3, damage=12),
        Character("Ranger", 2, 16, 10, {"STR": 3, "DEX": 4, "INT": 2}, attack_bonus=6, damage=10),
        Character("Cleric", 3, 15, 10, {"STR": 2, "DEX": 2, "INT": 4}, attack_bonus=4, damage=8),
        Character("Barbarian", 4, 26, 18, {"STR": 5, "DEX": 3, "INT": 1}, attack_bonus=8, damage=18),
    ]


def create_default_enemies() -> List[Character]:
    """Create default enemy group."""
    return [
        Character("Goblin", 0, 2, 1, {"DEX": 3}, attack_bonus=3, damage=6, role="enemy"),
        Character("Troll", 3, 6, 1, {"STR": 4, "DEX": 2}, attack_bonus=5, damage=8, role="enemy"),
        Character("Dragon", 4, 2, 2, {"STR": 6, "DEX": 6, "INT": 6}, attack_bonus=8, damage=12, role="enemy"),
    ]


def create_default_teammates() -> List[Character]:
    """Create 3 AI-controlled teammates for solo play."""
    return [
        Character("Aria", 0, 18, 16, {"STR": 3, "DEX": 4, "INT": 2}, attack_bonus=6, damage=10, role="teammate"),
        Character("Thorin", 1, 22, 18, {"STR": 5, "DEX": 2, "INT": 1}, attack_bonus=7, damage=14, role="teammate"),
        Character("Lyra", 2, 16, 14, {"STR": 2, "DEX": 3, "INT": 4}, attack_bonus=5, damage=9, role="teammate"),
    ]


def character_to_dict(char: Character) -> Dict:
    """Convert Character object to dictionary."""
    return {
        "name": char.name,
        "id": char.id,
        "hp": char.hp,
        "max_hp": char.max_hp,
        "ac": char.ac,
        "attributes": char.attributes,
        "attack_bonus": char.attack_bonus,
        "damage": char.damage,
        "role": char.role,
        "alive": char.alive,
    }


def get_combat_state(session_id: str, engine: CombatEngine) -> CombatState:
    """Generate current combat state."""
    winner = None
    if engine.is_battle_over():
        # Check if any players or teammates are alive (they're on the same side)
        players_and_teammates = [c for c in engine.state.players if c.role in ["player", "teammate"]]
        if any(p.alive for p in players_and_teammates):
            winner = "players"
        else:
            winner = "enemies"

    # Include teammates in the players list for the frontend (they're on the player's side)
    all_player_side = [p for p in engine.state.players if p.role in ["player", "teammate"]]

    return CombatState(
        session_id=session_id,
        round=engine.round,
        current_actor=engine.current_actor.name if engine.current_actor else None,
        players=[character_to_dict(p) for p in all_player_side],
        enemies=[character_to_dict(e) for e in engine.state.enemies],
        battle_over=engine.is_battle_over(),
        winner=winner,
    )


# ========== API Endpoints ==========
@router.post("/start")
async def start_combat(request: InitiateCombatRequest) -> Dict:
    """
    Initialize a new combat session with players and enemies.
    If not provided, uses default preset characters.
    """
    session_id = str(uuid.uuid4())

    # Create characters
    if request.players:
        players = [
            Character(p.name, i, p.hp, p.ac, p.attributes, p.attack_bonus, p.damage, p.role)
            for i, p in enumerate(request.players)
        ]
        # If only one player provided (solo play), add 3 AI teammates
        if len(players) == 1:
            teammates = create_default_teammates()
            # Add teammates to players list (they're on the player's side)
            players.extend(teammates)
            print(f"[COMBAT] Solo play detected: Added {len(teammates)} AI teammates")
    else:
        players = create_default_players()

    if request.enemies:
        enemies = [
            Character(e.name, i, e.hp, e.ac, e.attributes, e.attack_bonus, e.damage, e.role)
            for i, e in enumerate(request.enemies)
        ]
    else:
        enemies = create_default_enemies()

    # Create combat engine
    engine = CombatEngine(players, enemies)
    combat_sessions[session_id] = engine

    # Create parsers and AI components
    action_parsers[session_id] = ActionParser(engine)
    bot_parsers[session_id] = ActionParserBot(engine)
    bots[session_id] = DnDBot(engine)
    narrators[session_id] = DnDNarrator()

    # Set the first actor for the initial turn
    engine.next_turn()

    # Get initial state
    state = get_combat_state(session_id, engine)

    return {"session_id": session_id, "message": "Combat initiated! Roll for initiative!", "state": state.model_dump()}


@router.get("/state/{session_id}")
async def get_state(session_id: str) -> CombatState:
    """Get current combat state for a session."""
    if session_id not in combat_sessions:
        raise HTTPException(status_code=404, detail="Combat session not found")

    engine = combat_sessions[session_id]
    return get_combat_state(session_id, engine)


@router.post("/action/{session_id}")
async def player_action(session_id: str, request: PlayerActionRequest) -> ActionResponse:
    """
    Process a player action and advance combat.
    Handles both player and enemy turns automatically.
    """
    print(f"[COMBAT] ===== Action request received =====")
    print(f"[COMBAT] Session: {session_id}, Action: {request.action}")

    if session_id not in combat_sessions:
        print(f"[COMBAT] ERROR: Session {session_id} not found")
        raise HTTPException(status_code=404, detail="Combat session not found")

    engine = combat_sessions[session_id]
    parser = action_parsers[session_id]
    bot = bots[session_id]
    narrator = narrators[session_id]
    print(f"[COMBAT] Session found, engine initialized")

    if engine.is_battle_over():
        state = get_combat_state(session_id, engine)
        return ActionResponse(narrative="The battle has ended!", raw_result="Battle over", state=state)

    # Get current actor (or advance if none set)
    if not engine.current_actor:
        # No current actor - advance to first turn
        actor = engine.next_turn()
        # Skip dead actors
        while not actor.alive and not engine.is_battle_over():
            actor = engine.next_turn()
    else:
        # Use current actor (don't advance yet - we'll advance after processing)
        actor = engine.current_actor

    if engine.is_battle_over():
        state = get_combat_state(session_id, engine)
        return ActionResponse(narrative="The battle has ended!", raw_result="Battle over", state=state)

    # Validate turn ownership
    if actor.role == "player" or actor.role == "teammate":
        # It's a player or teammate turn - player controls all characters
        # Validate action was provided and not "enemy_turn"
        if request.action == "enemy_turn":
            raise HTTPException(status_code=400, detail="Cannot process enemy turn during player/teammate turn")
        if not request.action or request.action.strip() == "":
            raise HTTPException(status_code=400, detail="Action required for player/teammate turn")

        # Parse player action
        action = parser.parse(actor, request.action)
        if not action:
            raise HTTPException(status_code=400, detail="Could not parse action")

        # Execute action
        raw_result = engine.process_action(action)

        # Generate narrative (narrator.narrate is now async)
        narrative = await narrator.narrate(request.action, raw_result)

    else:
        # It's an enemy turn - process enemy action (ignore player-provided action text)
        print(f"[COMBAT] Processing enemy turn for {actor.name}")
        print(f"[COMBAT] Calling bot.decide_action()...")
        try:
            # Use await since decide_action is now async
            action = await bot.decide_action()
            print(f"[COMBAT] Enemy action received: {action}")
            if not action:
                print(f"[COMBAT] ERROR: Enemy bot returned None action")
                raise HTTPException(status_code=500, detail="AI failed to decide action")

            # Execute action
            print(f"[COMBAT] Executing enemy action...")
            raw_result = engine.process_action(action)
            print(f"[COMBAT] Action executed, result: {raw_result}")

            # Generate narrative (narrator.narrate is now async)
            narrative = await narrator.narrate(raw_result, raw_result)
            print(f"[COMBAT] Narrative generated: {narrative[:100]}...")
        except Exception as e:
            print(f"[COMBAT] ERROR in enemy turn processing: {e}")
            import traceback

            traceback.print_exc()
            raise

    # Advance to next turn after processing action
    engine.next_turn()

    # Skip dead actors in next turn
    while engine.current_actor and not engine.current_actor.alive and not engine.is_battle_over():
        engine.next_turn()

    # Get updated state
    state = get_combat_state(session_id, engine)

    return ActionResponse(narrative=narrative, raw_result=raw_result, state=state)


@router.delete("/session/{session_id}")
async def end_combat(session_id: str) -> Dict:
    """End a combat session and clean up resources."""
    if session_id not in combat_sessions:
        raise HTTPException(status_code=404, detail="Combat session not found")

    # Clean up all session data
    del combat_sessions[session_id]
    del action_parsers[session_id]
    del bot_parsers[session_id]
    del bots[session_id]
    del narrators[session_id]

    return {"message": "Combat session ended successfully"}
