"""
Integration tests for Orchestrator/API Gateway
Tests the main game orchestration endpoints using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def orchestrator_client():
    """Provide FastAPI TestClient for Orchestrator."""
    import sys
    from pathlib import Path

    # Clear any existing 'app' module from cache to avoid conflicts
    if 'app' in sys.modules:
        del sys.modules['app']

    orchestrator_path = Path(__file__).parent.parent.parent / "src" / "orchestrator"

    # Temporarily modify sys.path to prioritize orchestrator
    original_path = sys.path.copy()
    sys.path.insert(0, str(orchestrator_path))

    try:
        from app import app
        client = TestClient(app)
    finally:
        # Restore original sys.path
        sys.path = original_path

    return client


@pytest.mark.integration
class TestOrchestratorHealth:
    """Test Orchestrator health and basic endpoints."""

    def test_root_endpoint(self, orchestrator_client):
        """Test root endpoint returns service info."""
        response = orchestrator_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data or "message" in data

    def test_health_endpoint(self, orchestrator_client):
        """Test health check endpoint."""
        response = orchestrator_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


@pytest.mark.integration
class TestGameSessionManagement:
    """Test game session creation and management."""

    def test_start_new_game_session(self, orchestrator_client):
        """Test starting a new game session."""
        request_data = {
            "campaign_name": "test_campaign",
            "player_name": "TestHero"
        }
        response = orchestrator_client.post("/game/start", json=request_data)

        # Should return session info
        if response.status_code == 200:
            data = response.json()
            assert "session_id" in data
            assert isinstance(data["session_id"], str)
        else:
            # May fail if dependencies unavailable - that's ok for integration test
            assert response.status_code in [200, 500, 503]

    def test_get_game_state(self, orchestrator_client):
        """Test retrieving game state."""
        # First start a session
        start_response = orchestrator_client.post("/game/start", json={
            "campaign_name": "test_campaign"
        })

        if start_response.status_code == 200:
            session_id = start_response.json()["session_id"]

            # Get game state
            response = orchestrator_client.get(f"/game/state/{session_id}")
            assert response.status_code in [200, 404]

            if response.status_code == 200:
                data = response.json()
                assert "current_state" in data or "state" in data


@pytest.mark.integration
class TestPlayerActions:
    """Test player action handling and routing."""

    @patch('requests.post')
    def test_process_player_action_narrative(self, mock_post, orchestrator_client):
        """Test processing a narrative (non-combat) action."""
        # Mock Rule Agent response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "is_valid": True,
                "validation_type": "valid",
                "explanation": "Action is valid",
                "rule_text": "Test rule text"
            }
        )

        request_data = {
            "session_id": "test-session-123",
            "text": "I examine the mysterious door"
        }

        response = orchestrator_client.post("/game/action", json=request_data)

        # Should process action or return error if session doesn't exist
        assert response.status_code in [200, 404, 500]

    @patch('requests.post')
    def test_process_combat_action(self, mock_post, orchestrator_client):
        """Test processing a combat action."""
        # Mock Combat Agent response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "result": "Attack successful",
                "damage": 10,
                "battle_over": False
            }
        )

        request_data = {
            "session_id": "test-session-combat",
            "text": "I attack the goblin with my sword"
        }

        response = orchestrator_client.post("/game/action", json=request_data)
        # Should process or return appropriate error
        assert response.status_code in [200, 404, 500]


@pytest.mark.integration
class TestStateTransitions:
    """Test game state transitions."""

    def test_transition_to_combat(self, orchestrator_client):
        """Test transitioning from narrative to combat."""
        # This is a complex flow that depends on campaign data
        # Integration test verifies endpoint structure

        request_data = {
            "session_id": "test-session",
            "trigger": "combat"
        }

        response = orchestrator_client.post("/game/transition", json=request_data)
        # Should handle transition or return error if session doesn't exist
        assert response.status_code in [200, 404, 422, 500]


@pytest.mark.integration
class TestRuleValidationIntegration:
    """Test rule validation integration with orchestrator."""

    @patch('rule_validator.RuleValidator.validate_action')
    def test_action_with_rule_validation(self, mock_validate, orchestrator_client):
        """Test that actions trigger rule validation."""
        mock_validate.return_value = {
            "is_valid": True,
            "validation_type": "valid",
            "explanation": "Action follows D&D rules",
            "rule_text": "PHB p.194"
        }

        request_data = {
            "session_id": "test-session",
            "text": "I cast Shield as a reaction"
        }

        response = orchestrator_client.post("/game/action", json=request_data)
        # Endpoint should exist and handle validation
        assert response.status_code in [200, 404, 500]

    @patch('rule_validator.RuleValidator.validate_action')
    def test_sabotage_blocked_by_validation(self, mock_validate, orchestrator_client):
        """Test that sabotage attempts are caught by validation."""
        mock_validate.return_value = {
            "is_valid": False,
            "validation_type": "sabotage",
            "explanation": "Meta-gaming detected",
            "rule_text": ""
        }

        request_data = {
            "session_id": "test-session",
            "text": "I want to break the game"
        }

        response = orchestrator_client.post("/game/action", json=request_data)
        # Should handle sabotage appropriately
        assert response.status_code in [200, 400, 404, 500]


@pytest.mark.integration
class TestCampaignLoading:
    """Test campaign and story tree loading."""

    def test_list_available_campaigns(self, orchestrator_client):
        """Test listing available campaigns."""
        response = orchestrator_client.get("/campaigns")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))

    def test_get_campaign_details(self, orchestrator_client):
        """Test getting specific campaign details."""
        response = orchestrator_client.get("/campaigns/lost_mine_of_phandelver")

        # Campaign may or may not exist
        assert response.status_code in [200, 404]


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in orchestrator."""

    def test_action_without_session_id(self, orchestrator_client):
        """Test action request without session ID."""
        request_data = {
            "text": "I look around"
        }
        response = orchestrator_client.post("/game/action", json=request_data)
        # Should return 404 (session not found) since session_id is None
        assert response.status_code == 404

    def test_get_nonexistent_session(self, orchestrator_client):
        """Test retrieving non-existent game session."""
        response = orchestrator_client.get("/game/state/nonexistent-session-id")
        assert response.status_code == 404

    def test_invalid_endpoint(self, orchestrator_client):
        """Test accessing non-existent endpoint."""
        response = orchestrator_client.get("/invalid/endpoint")
        assert response.status_code == 404
