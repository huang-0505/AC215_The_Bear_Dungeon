"""
rule_validator_node.py

Graph node that validates player actions against D&D rules
using the Rule Agent RAG service.
"""

import os
import logging
import threading
from graph_state import DnDGameState
from rule_validator import RuleValidator

logger = logging.getLogger(__name__)

# Initialize rule validator (configured via env vars in graph.py)
_rule_validator: RuleValidator | None = None
_rv_lock = threading.Lock()


def get_rule_validator() -> RuleValidator:
    """Get or create the RuleValidator instance (thread-safe)."""
    global _rule_validator
    if _rule_validator is not None:
        return _rule_validator

    with _rv_lock:
        if _rule_validator is not None:
            return _rule_validator
        rule_agent_url = os.getenv("RULE_AGENT_URL", "http://localhost:9002")
        _rule_validator = RuleValidator(rule_agent_url=rule_agent_url)

    return _rule_validator


def rule_validator_node(state: DnDGameState) -> dict:
    """
    Validate the player's action against D&D rules.

    Reads: player_action
    Writes: validation_result, is_sabotage
    """
    player_action = state["player_action"]
    validator = get_rule_validator()

    # Build minimal game context for validation
    game_context = {
        "state_type": state["state_type"],
        "recent_actions": [],
        "in_combat": state["state_type"] == "combat",
    }

    validation = validator.validate_action(player_action, game_context)
    is_sabotage = validator.is_sabotage(validation)

    logger.info(f"Validation result: {validation.get('validation_type')}, sabotage: {is_sabotage}")

    return {
        "validation_result": validation,
        "is_sabotage": is_sabotage,
    }
