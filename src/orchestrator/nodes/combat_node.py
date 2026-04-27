"""
combat_node.py

Graph node that handles combat interactions via the Combat Agent service.
Manages combat start, actions, and post-combat narration transitions.

Multiplayer-aware: starts combat with all party members, but turn-by-turn
action handling still mirrors the legacy single-action shape — strict
initiative-ordered turns are introduced in Phase 5.
"""

import os
import logging
from typing import Optional
from uuid import uuid4

import requests
from langchain_core.messages import HumanMessage, AIMessage

from graph_state import DnDGameState
from models.player import PlayerCharacter

logger = logging.getLogger(__name__)

COMBAT_AGENT_URL = os.getenv("COMBAT_AGENT_URL", "http://localhost:9000")


def _get_enemy_pool() -> list[list[dict]]:
    """Define enemy pools for different combat encounters."""
    return [
        [
            {"name": "Goblin", "hp": 12, "ac": 13, "attributes": {"DEX": 3}, "attack_bonus": 3, "damage": 6, "role": "enemy"},
            {"name": "Kobold", "hp": 8, "ac": 12, "attributes": {"DEX": 2}, "attack_bonus": 2, "damage": 4, "role": "enemy"},
            {"name": "Orc", "hp": 15, "ac": 13, "attributes": {"STR": 3}, "attack_bonus": 4, "damage": 7, "role": "enemy"},
        ],
        [
            {"name": "Hobgoblin", "hp": 18, "ac": 15, "attributes": {"STR": 3, "DEX": 2}, "attack_bonus": 5, "damage": 8, "role": "enemy"},
            {"name": "Gnoll", "hp": 16, "ac": 14, "attributes": {"STR": 3, "DEX": 1}, "attack_bonus": 4, "damage": 7, "role": "enemy"},
            {"name": "Bugbear", "hp": 20, "ac": 14, "attributes": {"STR": 4, "DEX": 2}, "attack_bonus": 5, "damage": 9, "role": "enemy"},
        ],
        [
            {"name": "Troll", "hp": 16, "ac": 13, "attributes": {"STR": 4, "DEX": 2}, "attack_bonus": 5, "damage": 8, "role": "enemy"},
            {"name": "Ogre", "hp": 22, "ac": 13, "attributes": {"STR": 5, "DEX": 1}, "attack_bonus": 6, "damage": 10, "role": "enemy"},
            {"name": "Ettin", "hp": 20, "ac": 14, "attributes": {"STR": 5, "DEX": 1}, "attack_bonus": 6, "damage": 11, "role": "enemy"},
        ],
        [
            {"name": "Minotaur", "hp": 24, "ac": 16, "attributes": {"STR": 5, "DEX": 2}, "attack_bonus": 7, "damage": 12, "role": "enemy"},
            {"name": "Chimera", "hp": 20, "ac": 17, "attributes": {"STR": 5, "DEX": 3, "INT": 2}, "attack_bonus": 7, "damage": 13, "role": "enemy"},
            {"name": "Wyvern", "hp": 18, "ac": 18, "attributes": {"STR": 5, "DEX": 4}, "attack_bonus": 8, "damage": 14, "role": "enemy"},
        ],
        [
            {"name": "Dragon", "hp": 20, "ac": 20, "attributes": {"STR": 6, "DEX": 6, "INT": 6}, "attack_bonus": 8, "damage": 12, "role": "enemy"},
            {"name": "Lich", "hp": 18, "ac": 19, "attributes": {"STR": 3, "DEX": 3, "INT": 7}, "attack_bonus": 7, "damage": 15, "role": "enemy"},
            {"name": "Balor", "hp": 22, "ac": 20, "attributes": {"STR": 7, "DEX": 5, "INT": 4}, "attack_bonus": 9, "damage": 16, "role": "enemy"},
        ],
    ]


def _select_enemies(combat_count: int) -> list[dict]:
    """Select enemies from pool based on combat count (0-indexed)."""
    pools = _get_enemy_pool()
    pool_index = min(combat_count, len(pools) - 1)
    return pools[pool_index]


def _start_combat(players: list[dict], combat_count: int) -> dict:
    """Start a new combat session for the full party."""
    try:
        if not players:
            # Fallback: a default Fighter so combat can still start.
            fallback = PlayerCharacter.from_class(
                character_class="Fighter", character_name="Adventurer"
            )
            party_stats = [fallback.to_combat_stats()]
        else:
            party_stats = [PlayerCharacter.from_dict(p).to_combat_stats() for p in players]

        enemies = _select_enemies(combat_count)
        request_data = {"players": party_stats, "enemies": enemies}

        logger.info("Starting combat with party of %d vs %d enemies", len(party_stats), len(enemies))
        response = requests.post(f"{COMBAT_AGENT_URL}/combat/start", json=request_data, timeout=10)
        response.raise_for_status()
        start_data = response.json()

        # The combat agent's first round may begin with an enemy if their
        # initiative roll outranks every PC. Run those turns through before
        # we hand control back to the players.
        session_id = start_data.get("session_id")
        if session_id and start_data.get("state"):
            advanced = _auto_advance_enemy_turns(session_id, start_data)
            start_data["state"] = advanced.get("state", start_data["state"])

        return start_data
    except Exception as e:  # noqa: BLE001 - degraded combat start
        logger.error(f"Error starting combat: {e}")
        return {
            "session_id": str(uuid4()),
            "message": "⚔️ Combat begins! (Combat agent unavailable)",
            "state": {"battle_over": False},
        }


def _find_acting_player_id(players: list[dict], current_actor_name: Optional[str]) -> Optional[str]:
    """Map the combat agent's `current_actor` (a name) to a PlayerCharacter.player_id.

    The combat agent identifies actors by name only. We rely on `character_name`
    being unique within a party — collisions silently resolve to the first match.
    """
    if not current_actor_name:
        return None
    for p in players:
        if p.get("character_name") == current_actor_name:
            return p["player_id"]
    return None


def _is_enemy_turn(state_data: dict) -> bool:
    """True if `state_data.current_actor` matches an entry in state_data.enemies."""
    actor = state_data.get("current_actor")
    if not actor:
        return False
    return any(e.get("name") == actor for e in state_data.get("enemies") or [])


def _auto_advance_enemy_turns(combat_session_id: str, response: dict, max_iterations: int = 16) -> dict:
    """If the next actor is an enemy, run the combat agent until a PC's turn or
    battle end. Bounded iterations protect against pathological loops."""
    for _ in range(max_iterations):
        state_data = response.get("state", {}) or {}
        if state_data.get("battle_over"):
            return response
        if not _is_enemy_turn(state_data):
            return response
        logger.info("Auto-advancing enemy turn for %s", state_data.get("current_actor"))
        response = _send_combat_action(combat_session_id, "enemy_turn")
    logger.warning("Auto-advance hit max iterations for combat %s", combat_session_id)
    return response


def _send_combat_action(combat_session_id: str, action: str) -> dict:
    """Send an action to the combat agent."""
    try:
        response = requests.post(
            f"{COMBAT_AGENT_URL}/combat/action/{combat_session_id}",
            json={"action": action},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 400:
            state_data = _get_combat_state(combat_session_id)
            if state_data and state_data.get("battle_over"):
                return {"narrative": "The battle has ended!", "raw_result": "Battle over", "state": state_data}
        logger.error(f"Error in combat action: {e}")
        return {"narrative": f"Combat action error: {e}", "raw_result": "", "state": {"battle_over": True, "winner": "unknown"}}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error in combat action: {e}")
        return {"narrative": f"Combat action error: {e}", "raw_result": "", "state": {"battle_over": True, "winner": "unknown"}}


def _get_combat_state(combat_session_id: str) -> Optional[dict]:
    """Get combat state from the combat agent."""
    try:
        response = requests.get(f"{COMBAT_AGENT_URL}/combat/state/{combat_session_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error getting combat state: {e}")
        return None


def _call_narrator_for_post_combat(winner: str, combat_summary: str, story_summary: str) -> dict:
    """Call narrator for post-combat narration. Imports lazily to avoid circular deps."""
    from nodes.narrator_node import _call_narrator

    if winner == "players":
        prompt = (
            f"The heroes have emerged victorious from battle!\n\n"
            f"Combat Summary: {combat_summary}\n\n"
            f"As the dust settles, describe the aftermath, the sense of triumph, "
            f"and what lies ahead. Continue the adventure with vivid narration."
        )
    else:
        prompt = (
            f"The battle has ended, but not in the heroes' favor.\n\n"
            f"Combat Summary: {combat_summary}\n\n"
            f"Describe the aftermath and what happens next."
        )

    return _call_narrator(prompt, story_context=story_summary, generate_choices=True)


def combat_node(state: DnDGameState) -> dict:
    """
    Handle combat: start new combat or process combat action.
    Handles post-combat transition back to narration.

    Reads: player_action, state_type, combat_session_id, players,
           combat_count, combat_trigger, story_summary
    Writes: combat_response, combat_session_id, state_type, combat_count,
            response, choices, transition, narrator_response, messages, response_extras
    """
    current_state = state["state_type"]
    combat_session_id = state.get("combat_session_id")
    player_action = state["player_action"]
    players = state.get("players", [])
    combat_count = state.get("combat_count", 0)
    max_combats = state.get("max_combats", 5)
    story_summary = state.get("story_summary", "")
    narration_round = state.get("narration_round", 0)

    # --- CASE 1: Starting new combat (transitioning from narration) ---
    if current_state != "combat":
        trigger = state.get("combat_trigger", "player_requested")

        if trigger == "round_based":
            narration_round = narration_round + 1

        combat_start = _start_combat(players, combat_count)
        new_session_id = combat_start["session_id"]
        start_state = combat_start.get("state", {}) or {}
        acting_player_id = _find_acting_player_id(players, start_state.get("current_actor"))
        response_text = (
            f"⚔️ **Combat Encounter!**\n\n"
            f"{combat_start.get('message', 'Combat begins! Roll for initiative!')}"
        )

        return {
            "combat_response": combat_start,
            "combat_session_id": new_session_id,
            "state_type": "combat",
            "response": response_text,
            "choices": [],
            "transition": "narration -> combat",
            "combat_trigger": trigger,
            "narration_round": narration_round,
            "acting_player_id": acting_player_id,
            "messages": [AIMessage(content=response_text)],
            "response_extras": {
                "combat_session_id": new_session_id,
                "combat_state": start_state,
                "acting_player_id": acting_player_id,
            },
        }

    # --- CASE 2: Processing combat action ---
    if not combat_session_id:
        return {
            "response": "No active combat session.",
            "state_type": "narration",
            "choices": ["Continue exploring"],
        }

    combat_ended_keywords = {"combat ended", "combat end", "battle ended", "battle over"}
    if player_action.lower().strip() in combat_ended_keywords:
        combat_state = _get_combat_state(combat_session_id)
        if combat_state and combat_state.get("battle_over"):
            combat_response = {
                "narrative": "The battle has ended!",
                "raw_result": "Battle over",
                "state": combat_state,
            }
        else:
            combat_response = _send_combat_action(combat_session_id, player_action)
    else:
        combat_response = _send_combat_action(combat_session_id, player_action)

    # After the PC's turn, run any enemy turns until the next live PC's turn
    # or the battle ends. This keeps the multi-PC initiative cycle moving.
    combat_response = _auto_advance_enemy_turns(combat_session_id, combat_response)

    battle_over = combat_response.get("state", {}).get("battle_over", False)

    if not battle_over:
        next_state = combat_response.get("state", {}) or {}
        next_acting = _find_acting_player_id(players, next_state.get("current_actor"))
        return {
            "combat_response": combat_response,
            "response": combat_response.get("narrative", "Combat continues..."),
            "choices": [],
            "acting_player_id": next_acting,
            "messages": [
                HumanMessage(content=player_action),
                AIMessage(content=combat_response.get("narrative", "")),
            ],
            "response_extras": {
                "combat_state": next_state,
                "acting_player_id": next_acting,
            },
        }

    # --- CASE 3: Combat ended, transition back to narration ---
    new_combat_count = combat_count + 1
    winner = combat_response.get("state", {}).get("winner", "unknown")
    combat_summary = combat_response.get("narrative", "The battle concluded.")

    if new_combat_count >= max_combats:
        ending_text = (
            f"\U0001f389 **ADVENTURE COMPLETE!** \U0001f389\n\n"
            f"You have completed {new_combat_count} combat encounters! "
            f"Your journey has come to an end. Well done, adventurer!\n\n"
            f"*The adventure ends here. Thank you for playing!*"
        )
        return {
            "combat_response": combat_response,
            "combat_count": new_combat_count,
            "state_type": "game_over",
            "response": ending_text,
            "choices": [],
            "transition": "combat -> narration -> game_over",
            "is_ending": True,
            "ending_type": "victory",
            "acting_player_id": None,
            "messages": [AIMessage(content=ending_text)],
            "response_extras": {"combat_summary": combat_response},
        }

    try:
        narrator_result = _call_narrator_for_post_combat(winner, combat_summary, story_summary)
        post_combat_text = narrator_result.get("result", "")
        post_combat_choices = narrator_result.get("choices") or [
            "Take a moment to rest and recover.",
            "Search the area for useful items or clues.",
            "Continue exploring forward.",
            "Examine your surroundings more carefully.",
        ]
    except Exception as e:  # noqa: BLE001
        logger.error(f"Post-combat narration failed: {e}")
        post_combat_text = (
            f"As the dust settles, you stand victorious. The battle was fierce, "
            f"but your determination carried the day. The path ahead remains open."
        )
        post_combat_choices = [
            "Take a moment to rest and recover.",
            "Search the area for useful items or clues.",
            "Continue exploring forward.",
            "Examine your surroundings more carefully.",
        ]

    if not post_combat_text:
        post_combat_text = "The aftermath of battle surrounds you. What do you do next?"

    return {
        "combat_response": combat_response,
        "combat_count": new_combat_count,
        "state_type": "narration",
        "narrator_response": post_combat_text,
        "response": post_combat_text,
        "choices": post_combat_choices,
        "transition": "combat -> narration",
        "combat_session_id": None,
        "acting_player_id": None,
        "messages": [AIMessage(content=post_combat_text)],
        "response_extras": {"combat_summary": combat_response},
    }
