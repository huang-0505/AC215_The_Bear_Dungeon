"""
context_builder.py

Build game context for rule validation and agent routing.
Extracts relevant information from the game state tree.
"""

from typing import Dict, List, Optional
from game_state import GameStateTree, GameStateType, GameStateNode


class GameContextBuilder:
    """Build context for rule validation and agent communication"""

    @staticmethod
    def build_context(tree: GameStateTree) -> Dict:
        """
        Extract current game context for rule validation.

        Returns context like:
        {
            "state_type": "combat" | "narration",
            "characters": [...],  # Current party/enemies (if available)
            "combat_session_id": "...",  # If in combat
            "recent_actions": [...],  # Last 3-5 actions
            "current_conditions": {...}  # Status effects, HP, etc.
        }
        """
        current = tree.get_current()
        if not current:
            return {"state_type": "initial"}

        context = {"state_type": current.state_type.value, "recent_actions": [], "metadata": current.metadata}

        # Get recent action history
        path = tree.get_path_from_root()
        for node in path[-5:]:  # Last 5 nodes
            if node.player_action:
                context["recent_actions"].append(
                    {"action": node.player_action, "response": node.agent_response, "state_type": node.state_type.value}
                )

        # Add combat-specific context
        if current.state_type == GameStateType.COMBAT:
            context["combat_session_id"] = current.combat_session_id
            context["in_combat"] = True
        else:
            context["in_combat"] = False

        return context

    @staticmethod
    def build_agent_context(node: GameStateNode, tree: GameStateTree) -> Dict:
        """
        Build rich context for agent calls (narrator/combat).

        Includes:
        - Applicable D&D rules
        - Recent narrative/combat history
        - Current state information
        """
        context = {
            "current_state": node.state_type.value,
            "applicable_rules": node.applicable_rules,
            "recent_history": [],
        }

        # Get recent history from tree path
        path = tree.get_path_from_root()
        for hist_node in path[-3:]:  # Last 3 nodes
            if hist_node.agent_response:
                context["recent_history"].append(
                    {
                        "state_type": hist_node.state_type.value,
                        "action": hist_node.player_action,
                        "response": hist_node.agent_response,
                    }
                )

        return context

    @staticmethod
    def get_story_summary(tree: GameStateTree, max_nodes: int = 10) -> str:
        """
        Generate a text summary of the story so far.

        Useful for context when transitioning between states or
        providing the narrator with background.
        """
        path = tree.get_path_from_root()
        summary_parts = []

        for node in path[-max_nodes:]:
            if node.state_type == GameStateType.NARRATION and node.agent_response:
                summary_parts.append(f"[Narration] {node.agent_response}")
            elif node.state_type == GameStateType.COMBAT and node.agent_response:
                summary_parts.append(f"[Combat] {node.agent_response}")

        return "\n\n".join(summary_parts)
