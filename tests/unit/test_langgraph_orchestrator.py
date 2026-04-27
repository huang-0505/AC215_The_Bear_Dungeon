"""
test_langgraph_orchestrator.py

Unit tests for the LangGraph-based D&D game orchestrator.
Tests the graph structure, routing logic, and node behavior
without requiring any external services (all mocked).
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add orchestrator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src/orchestrator"))


# ========== Graph State Tests ==========
@pytest.mark.unit
class TestDnDGameState:
    """Test the shared state definition."""

    def test_state_has_required_fields(self):
        from graph_state import DnDGameState

        # Verify all required fields exist in the TypedDict
        annotations = DnDGameState.__annotations__
        required_fields = [
            "player_action", "session_id", "state_type",
            "narration_round", "combat_count", "max_combats",
            "combat_rounds", "validation_result", "is_sabotage",
            "narrator_response", "choices", "combat_response",
            "combat_session_id", "combat_trigger", "messages",
            "story_summary", "response", "transition",
        ]
        for field in required_fields:
            assert field in annotations, f"Missing field: {field}"


# ========== Router Node Tests ==========
@pytest.mark.unit
class TestRouterNode:
    """Test the routing logic."""

    def _make_state(self, **overrides):
        """Create a minimal valid state dict."""
        base = {
            "player_action": "I look around",
            "session_id": "test-session",
            "state_type": "narration",
            "narration_round": 0,
            "combat_count": 0,
            "max_combats": 5,
            "combat_rounds": [3, 5, 10, 15],
            "is_sabotage": False,
            "combat_trigger": "none",
        }
        base.update(overrides)
        return base

    def test_routes_sabotage_to_reject(self):
        from nodes.router_node import router_node
        state = self._make_state(is_sabotage=True)
        assert router_node(state) == "reject"

    def test_routes_game_over_state(self):
        from nodes.router_node import router_node
        state = self._make_state(state_type="game_over")
        assert router_node(state) == "game_over"

    def test_routes_max_combats_to_game_over(self):
        from nodes.router_node import router_node
        state = self._make_state(combat_count=5, max_combats=5)
        assert router_node(state) == "game_over"

    def test_routes_combat_state_to_combat(self):
        from nodes.router_node import router_node
        state = self._make_state(state_type="combat")
        assert router_node(state) == "combat"

    def test_routes_round_based_combat_trigger(self):
        from nodes.router_node import router_node
        # narration_round=2, next_round=3 which is in combat_rounds
        state = self._make_state(narration_round=2)
        result = router_node(state)
        assert result == "combat"
        assert state["combat_trigger"] == "round_based"

    def test_routes_player_requested_combat(self):
        from nodes.router_node import router_node
        state = self._make_state(player_action="enter combat")
        result = router_node(state)
        assert result == "combat"
        assert state["combat_trigger"] == "player_requested"

    def test_routes_normal_action_to_narrator(self):
        from nodes.router_node import router_node
        state = self._make_state(player_action="I search the room")
        assert router_node(state) == "narrator"

    def test_combat_keyword_case_insensitive(self):
        from nodes.router_node import router_node
        state = self._make_state(player_action="  COMBAT  ")
        assert router_node(state) == "combat"


# ========== Rule Validator Node Tests ==========
@pytest.mark.unit
class TestRuleValidatorNode:
    """Test the rule validation node."""

    @patch("nodes.rule_validator_node.get_rule_validator")
    def test_validates_action(self, mock_get_validator):
        from nodes.rule_validator_node import rule_validator_node

        mock_validator = MagicMock()
        mock_validator.validate_action.return_value = {
            "is_valid": True,
            "validation_type": "valid",
            "rule_text": "Attack rules apply",
            "explanation": "Valid action",
        }
        mock_validator.is_sabotage.return_value = False
        mock_get_validator.return_value = mock_validator

        state = {
            "player_action": "I attack the goblin",
            "state_type": "combat",
        }
        result = rule_validator_node(state)

        assert result["is_sabotage"] is False
        assert result["validation_result"]["is_valid"] is True
        mock_validator.validate_action.assert_called_once()

    @patch("nodes.rule_validator_node.get_rule_validator")
    def test_detects_sabotage(self, mock_get_validator):
        from nodes.rule_validator_node import rule_validator_node

        mock_validator = MagicMock()
        mock_validator.validate_action.return_value = {
            "is_valid": False,
            "validation_type": "sabotage",
            "explanation": "Meta-gaming detected",
        }
        mock_validator.is_sabotage.return_value = True
        mock_get_validator.return_value = mock_validator

        state = {
            "player_action": "kill the boss right now",
            "state_type": "narration",
        }
        result = rule_validator_node(state)

        assert result["is_sabotage"] is True


# ========== Reject Node Tests ==========
@pytest.mark.unit
class TestRejectNode:
    """Test the sabotage rejection node."""

    def test_sanitizes_player_action(self):
        from nodes.reject_node import reject_node

        state = {
            "player_action": '<script>alert("xss")</script>',
            "validation_result": {"validation_type": "sabotage"},
        }
        result = reject_node(state)

        assert "<script>" not in result["response"]
        assert "&lt;script&gt;" in result["response"]
        assert result["response_extras"]["error"] == "invalid_action"

    def test_truncates_long_input(self):
        from nodes.reject_node import reject_node

        state = {
            "player_action": "A" * 500,
            "validation_result": {},
        }
        result = reject_node(state)

        # Should be truncated to MAX_ECHO_LENGTH (200)
        assert len(result["response"]) < 500


# ========== Game Over Node Tests ==========
@pytest.mark.unit
class TestGameOverNode:
    """Test the game over node."""

    def test_victory_ending(self):
        from nodes.game_over_node import game_over_node

        state = {"combat_count": 5, "ending_type": "victory"}
        result = game_over_node(state)

        assert result["state_type"] == "game_over"
        assert result["is_ending"] is True
        assert "ADVENTURE COMPLETE" in result["response"]
        assert result["choices"] == []

    def test_defeat_ending(self):
        from nodes.game_over_node import game_over_node

        state = {"combat_count": 3, "ending_type": "defeat"}
        result = game_over_node(state)

        assert "GAME OVER" in result["response"]

    def test_neutral_ending(self):
        from nodes.game_over_node import game_over_node

        state = {"combat_count": 2, "ending_type": "neutral"}
        result = game_over_node(state)

        assert "THE END" in result["response"]


# ========== Response Builder Node Tests ==========
@pytest.mark.unit
class TestResponseBuilderNode:
    """Test the response builder node."""

    def test_adds_common_fields(self):
        from nodes.response_builder_node import response_builder_node

        state = {
            "narration_round": 5,
            "combat_count": 2,
            "max_combats": 5,
            "is_ending": False,
            "ending_type": None,
            "transition": "narration -> combat",
            "validation_result": {"is_valid": True},
            "response_extras": {},
        }
        result = response_builder_node(state)
        extras = result["response_extras"]

        assert extras["narration_round"] == 5
        assert extras["combat_count"] == 2
        assert extras["max_combats"] == 5
        assert extras["transition"] == "narration -> combat"
        assert extras["validation"]["is_valid"] is True


# ========== Narrator Node Tests ==========
@pytest.mark.unit
class TestNarratorNode:
    """Test the narrator node."""

    @patch("nodes.narrator_node._detect_combat_trigger", return_value=False)
    @patch("nodes.narrator_node._call_narrator")
    def test_calls_narrator_and_increments_round(self, mock_narrator, mock_trigger):
        from nodes.narrator_node import narrator_node

        mock_narrator.return_value = {
            "result": "You enter a dark cave...",
            "choices": ["Go left", "Go right", "Turn back"],
        }

        state = {
            "player_action": "I enter the cave",
            "validation_result": {"rule_text": ""},
            "campaign_id": None,
            "current_story_node_id": None,
            "story_summary": "",
            "narration_round": 2,
        }
        result = narrator_node(state)

        assert result["narration_round"] == 3  # Incremented
        assert result["narrator_response"] == "You enter a dark cave..."
        assert len(result["choices"]) == 3
        assert result["state_type"] == "narration"
        assert result["combat_trigger"] == "none"

    @patch("nodes.narrator_node._detect_combat_trigger", return_value=True)
    @patch("nodes.narrator_node._call_narrator")
    def test_detects_ai_combat_trigger(self, mock_narrator, mock_trigger):
        from nodes.narrator_node import narrator_node

        mock_narrator.return_value = {
            "result": "Goblins ambush you from the shadows!",
            "choices": ["Fight", "Flee"],
        }

        state = {
            "player_action": "I walk into the forest",
            "validation_result": {},
            "campaign_id": None,
            "current_story_node_id": None,
            "story_summary": "",
            "narration_round": 1,
            "is_ending": False,
        }
        result = narrator_node(state)

        assert result["combat_trigger"] == "ai_detected"


# ========== Combat Node Tests ==========
@pytest.mark.unit
class TestCombatNode:
    """Test the combat node."""

    @patch("nodes.combat_node._start_combat")
    def test_starts_new_combat_from_narration(self, mock_start):
        from nodes.combat_node import combat_node

        mock_start.return_value = {
            "session_id": "combat-123",
            "message": "Combat begins!",
            "state": {"battle_over": False},
        }

        state = {
            "player_action": "enter combat",
            "state_type": "narration",  # Not in combat yet
            "combat_session_id": None,
            "character_class": "Fighter",
            "combat_count": 0,
            "max_combats": 5,
            "story_summary": "",
            "narration_round": 2,
            "combat_rounds": [3, 5, 10, 15],
            "combat_trigger": "player_requested",
        }
        result = combat_node(state)

        assert result["state_type"] == "combat"
        assert result["combat_session_id"] == "combat-123"
        assert result["transition"] == "narration -> combat"
        mock_start.assert_called_once_with("Fighter", 0)

    @patch("nodes.combat_node._send_combat_action")
    def test_processes_combat_action(self, mock_action):
        from nodes.combat_node import combat_node

        mock_action.return_value = {
            "narrative": "You slash the goblin!",
            "raw_result": "Hit for 10 damage",
            "state": {"battle_over": False},
        }

        state = {
            "player_action": "I attack the goblin",
            "state_type": "combat",
            "combat_session_id": "combat-123",
            "character_class": "Fighter",
            "combat_count": 0,
            "max_combats": 5,
            "story_summary": "",
            "narration_round": 3,
            "combat_rounds": [3, 5, 10, 15],
            "combat_trigger": "none",
        }
        result = combat_node(state)

        assert "You slash the goblin!" in result["response"]
        assert result.get("state_type") is None or result.get("state_type") == "combat"

    @patch("nodes.combat_node._call_narrator_for_post_combat")
    @patch("nodes.combat_node._send_combat_action")
    def test_transitions_to_narration_after_combat_ends(self, mock_action, mock_narrator):
        from nodes.combat_node import combat_node

        mock_action.return_value = {
            "narrative": "The goblin falls!",
            "raw_result": "Victory",
            "state": {"battle_over": True, "winner": "players"},
        }
        mock_narrator.return_value = {
            "result": "You stand victorious...",
            "choices": ["Rest", "Explore", "Loot"],
        }

        state = {
            "player_action": "I finish the goblin",
            "state_type": "combat",
            "combat_session_id": "combat-123",
            "character_class": "Fighter",
            "combat_count": 0,
            "max_combats": 5,
            "story_summary": "",
            "narration_round": 3,
            "combat_rounds": [3, 5, 10, 15],
            "combat_trigger": "none",
        }
        result = combat_node(state)

        assert result["state_type"] == "narration"
        assert result["combat_count"] == 1
        assert result["transition"] == "combat -> narration"
        assert result["combat_session_id"] is None  # Cleared

    @patch("nodes.combat_node._send_combat_action")
    def test_game_over_after_max_combats(self, mock_action):
        from nodes.combat_node import combat_node

        mock_action.return_value = {
            "narrative": "Final victory!",
            "state": {"battle_over": True, "winner": "players"},
        }

        state = {
            "player_action": "I attack",
            "state_type": "combat",
            "combat_session_id": "combat-123",
            "character_class": "Fighter",
            "combat_count": 4,  # Will become 5 = max
            "max_combats": 5,
            "story_summary": "",
            "narration_round": 10,
            "combat_rounds": [3, 5, 10, 15],
            "combat_trigger": "none",
        }
        result = combat_node(state)

        assert result["state_type"] == "game_over"
        assert result["is_ending"] is True
        assert result["combat_count"] == 5


# ========== Summarizer Node Tests ==========
@pytest.mark.unit
class TestSummarizerNode:
    """Test the summarizer node."""

    def test_no_summarize_on_non_interval(self):
        from nodes.summarizer_node import summarizer_node

        state = {
            "narration_round": 3,
            "messages": [],
            "story_summary": "",
        }
        result = summarizer_node(state)

        # Should return empty or minimal update
        assert "story_summary" not in result or result.get("story_summary", "") == ""

    @patch("nodes.summarizer_node._summarize_with_llm")
    def test_summarizes_at_interval(self, mock_summarize):
        from nodes.summarizer_node import summarizer_node

        mock_summarize.return_value = "Summary of events so far."

        # Create mock messages
        mock_messages = []
        for i in range(10):
            msg = MagicMock()
            msg.content = f"Event {i}"
            msg.type = "human" if i % 2 == 0 else "ai"
            mock_messages.append(msg)

        state = {
            "narration_round": 5,  # Multiple of SUMMARIZE_INTERVAL
            "messages": mock_messages,
            "story_summary": "",
        }
        result = summarizer_node(state)

        assert result["story_summary"] == "Summary of events so far."
        mock_summarize.assert_called_once()


# ========== Integration: Full Graph Tests ==========
@pytest.mark.unit
class TestGraphStructure:
    """Test that the graph compiles and has correct structure."""

    @patch("nodes.narrator_node._get_llm_client", return_value=(None, None, None))
    @patch("nodes.narrator_node._get_openai_client", return_value=None)
    def test_graph_compiles(self, mock_openai, mock_llm):
        """Verify the graph compiles without errors."""
        from graph import _build_graph
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3

        builder = _build_graph()
        # Compile with in-memory SQLite for testing
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        graph = builder.compile(checkpointer=checkpointer)

        assert graph is not None
        conn.close()

    @patch("nodes.narrator_node._get_llm_client", return_value=(None, None, None))
    @patch("nodes.narrator_node._get_openai_client", return_value=None)
    def test_graph_has_all_nodes(self, mock_openai, mock_llm):
        """Verify all expected nodes are registered."""
        from graph import _build_graph

        builder = _build_graph()
        node_names = set(builder.nodes.keys())
        expected = {"rule_validator", "narrator", "combat", "game_over",
                    "response_builder", "reject", "summarizer"}
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"
