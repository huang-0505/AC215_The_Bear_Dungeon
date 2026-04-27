"""
router_node.py

Conditional routing function that determines which agent node to invoke next.
This replaces the if/else routing logic in the old app.py.
"""

import logging
from graph_state import DnDGameState

logger = logging.getLogger(__name__)

# Combat entry keywords the player can type
COMBAT_KEYWORDS = {
    "enter combat", "combat", "fight",
    "\u2694\ufe0f enter combat", "\u2694\ufe0f combat",
}


def router_node(state: DnDGameState) -> str:
    """
    Route to the appropriate agent node based on game state and validation.

    Also sets combat_trigger in state so combat_node knows why it was invoked.
    Returns one of: "reject", "game_over", "combat", "narrator"
    """
    # 1. Sabotage -> reject
    if state.get("is_sabotage", False):
        logger.warning("Sabotage detected, routing to reject")
        return "reject"

    # 2. Game over check
    current_state = state["state_type"]
    if current_state == "game_over":
        return "game_over"

    combat_count = state.get("combat_count", 0)
    max_combats = state.get("max_combats", 5)
    if combat_count >= max_combats:
        return "game_over"

    # 3. Currently in combat -> combat node (continuing existing combat)
    if current_state == "combat":
        return "combat"

    # 4. Narration state: check for combat triggers
    narration_round = state.get("narration_round", 0)
    combat_rounds = state.get("combat_rounds", [3, 5, 10, 15])

    # Round-based forced combat
    next_round = narration_round + 1
    if next_round in combat_rounds and combat_count < max_combats:
        logger.info("Round-based combat trigger at round %d", next_round)
        # Set combat_trigger in state so combat_node knows the reason
        state["combat_trigger"] = "round_based"
        return "combat"

    # Player requested combat
    player_action = state.get("player_action", "").lower().strip()
    if player_action in COMBAT_KEYWORDS:
        logger.info("Player requested combat")
        state["combat_trigger"] = "player_requested"
        return "combat"

    # 5. Default: narration
    return "narrator"
