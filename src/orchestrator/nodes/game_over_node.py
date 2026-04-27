"""
game_over_node.py

Graph node that handles game ending states.
"""

import logging
from langchain_core.messages import AIMessage

from graph_state import DnDGameState

logger = logging.getLogger(__name__)


def game_over_node(state: DnDGameState) -> dict:
    """
    Handle game over state. Generates ending text based on context.

    Reads: combat_count, max_combats, ending_type
    Writes: state_type, response, choices, is_ending, ending_type, messages
    """
    combat_count = state.get("combat_count", 0)
    ending_type = state.get("ending_type", "victory")

    if ending_type == "victory":
        ending_text = (
            f"\U0001f389 **ADVENTURE COMPLETE!** \U0001f389\n\n"
            f"You have completed {combat_count} combat encounters! "
            f"Your journey has come to an end. Well done, adventurer!\n\n"
            f"*The adventure ends here. Thank you for playing!*"
        )
    elif ending_type == "defeat":
        ending_text = (
            f"\U0001f480 **GAME OVER** \U0001f480\n\n"
            f"Your adventure has come to a tragic end after {combat_count} battles. "
            f"Perhaps another hero will succeed where you have fallen.\n\n"
            f"*The adventure ends here.*"
        )
    else:
        ending_text = (
            f"**THE END**\n\n"
            f"Your adventure has reached its conclusion after {combat_count} encounters. "
            f"The story ends here, but the memories remain.\n\n"
            f"*Thank you for playing!*"
        )

    return {
        "state_type": "game_over",
        "response": ending_text,
        "choices": [],
        "is_ending": True,
        "ending_type": ending_type,
        "messages": [AIMessage(content=ending_text)],
    }
