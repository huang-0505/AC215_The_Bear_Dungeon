"""
Integration tests for Rule Agent API
Tests the rule validation and retrieval endpoints using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def rule_agent_client():
    """Provide FastAPI TestClient for Rule Agent."""
    # Import the Rule Agent app
    import sys
    from pathlib import Path

    # Clear any existing 'app' module from cache to avoid conflicts
    if 'app' in sys.modules:
        del sys.modules['app']

    rule_agent_path = Path(__file__).parent.parent.parent / "src" / "rule_agent"

    # Temporarily modify sys.path to prioritize rule_agent
    original_path = sys.path.copy()
    sys.path.insert(0, str(rule_agent_path))

    try:
        from app import app
        client = TestClient(app)
    finally:
        # Restore original sys.path
        sys.path = original_path

    return client


@pytest.mark.integration
class TestRuleAgentHealth:
    """Test Rule Agent health and basic endpoints."""

    def test_root_endpoint(self, rule_agent_client):
        """Test root endpoint returns service info."""
        response = rule_agent_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "D&D Rule Agent" in data["service"]

    def test_health_endpoint(self, rule_agent_client):
        """Test health check endpoint."""
        response = rule_agent_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


@pytest.mark.integration
class TestRuleValidation:
    """Test rule validation endpoints."""

    def test_validate_action_basic(self, rule_agent_client):
        """Test basic action validation."""
        request_data = {
            "user_input": "I attack the goblin with my sword",
            "context": {"in_combat": True}
        }
        response = rule_agent_client.post("/validate", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "is_valid" in data
        assert "validation_type" in data
        assert "explanation" in data
        assert "rule_text" in data
        assert isinstance(data["is_valid"], bool)

    def test_validate_sabotage_detection(self, rule_agent_client):
        """Test sabotage/meta-gaming detection."""
        request_data = {
            "user_input": "I want to sabotage the campaign and destroy everything",
            "context": {}
        }
        response = rule_agent_client.post("/validate", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # Should detect sabotage
        assert data["is_valid"] is False
        assert data["validation_type"] == "sabotage"
        assert "sabotage" in data["explanation"].lower() or "meta-gaming" in data["explanation"].lower()

    def test_validate_normal_combat_action(self, rule_agent_client):
        """Test validation of normal combat action."""
        request_data = {
            "user_input": "I cast fireball at the enemies",
            "context": {"player_class": "Wizard", "spell_slots": 3}
        }
        response = rule_agent_client.post("/validate", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # Should return validation result
        assert "is_valid" in data
        assert "validation_type" in data

    def test_validate_empty_input(self, rule_agent_client):
        """Test validation with empty input."""
        request_data = {
            "user_input": "",
            "context": {}
        }
        response = rule_agent_client.post("/validate", json=request_data)
        # Should still return 200 but handle gracefully
        assert response.status_code == 200


@pytest.mark.integration
class TestRuleRetrieval:
    """Test rule retrieval endpoints."""

    def test_retrieve_rules_basic(self, rule_agent_client):
        """Test basic rule retrieval."""
        request_data = {
            "query": "How does spell casting work?",
            "n_results": 5
        }
        response = rule_agent_client.post("/retrieve_rules", json=request_data)

        # May fail if ChromaDB is not available, which is ok for integration tests
        if response.status_code == 200:
            data = response.json()
            assert "rules" in data
            assert isinstance(data["rules"], str)
        else:
            # ChromaDB unavailable is acceptable in integration test
            assert response.status_code == 503

    def test_retrieve_rules_custom_count(self, rule_agent_client):
        """Test rule retrieval with custom result count."""
        request_data = {
            "query": "advantage and disadvantage",
            "n_results": 3
        }
        response = rule_agent_client.post("/retrieve_rules", json=request_data)

        # May fail if ChromaDB is not available
        if response.status_code == 200:
            data = response.json()
            assert "rules" in data

    def test_retrieve_rules_combat_mechanics(self, rule_agent_client):
        """Test retrieving combat-specific rules."""
        request_data = {
            "query": "attack roll and armor class",
            "n_results": 5
        }
        response = rule_agent_client.post("/retrieve_rules", json=request_data)

        # Should return results if ChromaDB is available
        assert response.status_code in [200, 503]


@pytest.mark.integration
class TestRuleAgentErrorHandling:
    """Test error handling in Rule Agent."""

    def test_validate_missing_user_input(self, rule_agent_client):
        """Test validation with missing required field."""
        request_data = {
            "context": {}
        }
        response = rule_agent_client.post("/validate", json=request_data)
        # FastAPI should return 422 for validation error
        assert response.status_code == 422

    def test_retrieve_rules_missing_query(self, rule_agent_client):
        """Test rule retrieval with missing query."""
        request_data = {
            "n_results": 5
        }
        response = rule_agent_client.post("/retrieve_rules", json=request_data)
        # FastAPI should return 422 for validation error
        assert response.status_code == 422

    def test_invalid_endpoint(self, rule_agent_client):
        """Test accessing non-existent endpoint."""
        response = rule_agent_client.get("/nonexistent")
        assert response.status_code == 404
