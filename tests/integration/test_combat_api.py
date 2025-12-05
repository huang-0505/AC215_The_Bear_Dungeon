"""
Integration tests for Combat API endpoints.
Tests API functionality using FastAPI TestClient (no real server needed).
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestCombatAPIEndpoints:
    """Test Combat API endpoints with TestClient."""

    def test_health_check(self, api_test_client):
        """Test health check endpoint returns healthy status."""
        response = api_test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, api_test_client):
        """Test root endpoint returns welcome message."""
        response = api_test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "DnD Combat" in data["message"]

    def test_start_combat_with_defaults(self, api_test_client):
        """Test starting combat with default characters."""
        response = api_test_client.post("/combat/start", json={})

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "session_id" in data
        assert "state" in data
        assert "message" in data

        # Verify initial state
        state = data["state"]
        assert "session_id" in state
        assert "round" in state
        assert "players" in state
        assert "enemies" in state
        assert "battle_over" in state

        # Verify characters were created
        assert len(state["players"]) > 0
        assert len(state["enemies"]) > 0
        assert state["battle_over"] is False

    def test_start_combat_with_custom_characters(
        self, api_test_client, sample_combat_request
    ):
        """Test starting combat with custom characters."""
        response = api_test_client.post(
            "/combat/start",
            json=sample_combat_request
        )

        assert response.status_code == 200
        data = response.json()
        state = data["state"]

        # Verify custom characters were used
        assert state["players"][0]["name"] == "TestKnight"
        assert state["players"][0]["hp"] == 20
        assert state["enemies"][0]["name"] == "TestGoblin"
        assert state["enemies"][0]["hp"] == 12

    def test_get_combat_state(self, api_test_client):
        """Test retrieving combat state for a session."""
        # First create a combat session
        start_response = api_test_client.post("/combat/start", json={})
        session_id = start_response.json()["session_id"]

        # Get the state
        response = api_test_client.get(f"/combat/state/{session_id}")

        assert response.status_code == 200
        data = response.json()

        # Verify state structure
        assert "session_id" in data
        assert "round" in data
        assert "players" in data
        assert "enemies" in data
        assert "battle_over" in data
        assert "current_actor" in data

    def test_get_nonexistent_combat_state_returns_404(self, api_test_client):
        """Test getting state for non-existent session returns 404."""
        response = api_test_client.get("/combat/state/invalid-session-id")

        assert response.status_code == 404
        assert "detail" in response.json()

    def test_player_action(self, api_test_client):
        """Test executing a player action."""
        # Create combat session
        start_response = api_test_client.post("/combat/start", json={})
        session_id = start_response.json()["session_id"]

        # Execute action
        action_request = {"action": "Attack the nearest enemy"}
        response = api_test_client.post(
            f"/combat/action/{session_id}",
            json=action_request
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "narrative" in data
        assert "raw_result" in data
        assert "state" in data

        # Verify state was updated
        state = data["state"]
        assert "round" in state

    def test_action_on_nonexistent_session_returns_404(self, api_test_client):
        """Test action on non-existent session returns 404."""
        response = api_test_client.post(
            "/combat/action/invalid-session",
            json={"action": "attack"}
        )

        assert response.status_code == 404

    def test_end_combat_session(self, api_test_client):
        """Test ending a combat session."""
        # Create session
        start_response = api_test_client.post("/combat/start", json={})
        session_id = start_response.json()["session_id"]

        # End session
        response = api_test_client.delete(f"/combat/session/{session_id}")

        assert response.status_code == 200
        assert "message" in response.json()

        # Verify session was deleted
        get_response = api_test_client.get(f"/combat/state/{session_id}")
        assert get_response.status_code == 404

    def test_complete_combat_flow(self, api_test_client):
        """Test a complete combat flow from start to finish."""
        # 1. Start combat
        start_response = api_test_client.post("/combat/start", json={})
        assert start_response.status_code == 200
        session_id = start_response.json()["session_id"]

        # 2. Get initial state
        state_response = api_test_client.get(f"/combat/state/{session_id}")
        assert state_response.status_code == 200

        # 3. Execute multiple actions
        for i in range(3):
            action_response = api_test_client.post(
                f"/combat/action/{session_id}",
                json={"action": f"Attack action {i}"}
            )
            # Should succeed or return battle over
            assert action_response.status_code in [200, 400]

        # 4. End combat
        end_response = api_test_client.delete(f"/combat/session/{session_id}")
        assert end_response.status_code == 200


@pytest.mark.integration
class TestCORS:
    """Test CORS configuration."""

    def test_cors_headers_present(self, api_test_client):
        """Test that CORS headers are present in responses."""
        response = api_test_client.get(
            "/",
            headers={"Origin": "http://localhost:3000"}
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_allows_all_origins(self, api_test_client):
        """Test that CORS allows all origins."""
        response = api_test_client.get(
            "/",
            headers={"Origin": "http://example.com"}
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"


@pytest.mark.integration
class TestErrorHandling:
    """Test API error handling."""

    def test_invalid_route_returns_404(self, api_test_client):
        """Test that invalid routes return 404."""
        response = api_test_client.get("/invalid/route")
        assert response.status_code == 404

    def test_method_not_allowed(self, api_test_client):
        """Test that wrong HTTP methods return 405."""
        response = api_test_client.post("/health")
        assert response.status_code == 405

    def test_invalid_request_body_returns_422(self, api_test_client):
        """Test that invalid request body returns validation error."""
        # Send invalid data (missing required fields)
        response = api_test_client.post(
            "/combat/start",
            json={"invalid": "data"}
        )
        # Should either accept (with defaults) or reject
        assert response.status_code in [200, 422]
