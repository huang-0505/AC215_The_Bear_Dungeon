"""
reject_node.py

Graph node that handles sabotage/invalid action rejection.
"""

import html
from graph_state import DnDGameState

# Max length of player action to echo back
MAX_ECHO_LENGTH = 200


def reject_node(state: DnDGameState) -> dict:
    """
    Reject a sabotage or meta-gaming attempt.

    Reads: player_action, validation_result
    Writes: response, choices, response_extras
    """
    player_action = state["player_action"]
    validation = state.get("validation_result", {})

    # Sanitize and truncate player action before echoing
    sanitized = html.escape(player_action[:MAX_ECHO_LENGTH])

    response_text = (
        f"Your input: '{sanitized}'\n\n"
        "This appears to be a meta-game or sabotage attempt. "
        "Please provide an in-character action that follows D&D rules."
    )

    return {
        "response": response_text,
        "choices": [],
        "response_extras": {
            "error": "invalid_action",
            "validation": validation,
        },
    }
