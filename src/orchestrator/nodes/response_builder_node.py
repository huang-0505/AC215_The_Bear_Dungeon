"""
response_builder_node.py

Graph node that builds the final response dict for the frontend.
This node runs after narrator/combat/game_over to ensure consistent response format.
"""

import logging
from graph_state import DnDGameState

logger = logging.getLogger(__name__)


def response_builder_node(state: DnDGameState) -> dict:
    """
    Build the final response to return to the frontend.
    Ensures consistent response format regardless of which agent handled the action.

    Reads: all state fields
    Writes: response_extras (merges additional frontend-facing data)
    """
    extras = state.get("response_extras", {})

    # Add common fields the frontend expects
    extras["narration_round"] = state.get("narration_round", 0)
    extras["combat_count"] = state.get("combat_count", 0)
    extras["max_combats"] = state.get("max_combats", 5)
    extras["is_ending"] = state.get("is_ending", False)
    extras["ending_type"] = state.get("ending_type")

    if state.get("transition"):
        extras["transition"] = state["transition"]

    if state.get("validation_result"):
        extras["validation"] = state["validation_result"]

    return {"response_extras": extras}
