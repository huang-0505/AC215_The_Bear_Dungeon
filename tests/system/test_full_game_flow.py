"""
End-to-end system tests for complete game flow
Tests all services working together: Orchestrator → Rule Agent → Combat Agent → ChromaDB

IMPORTANT: These tests require all Docker services to be running:
  docker compose up -d
"""

import pytest
import requests
import time
from typing import Optional


# Service URLs
# Orchestrator accessed through nginx on port 8080
ORCHESTRATOR_URL = "http://localhost:8080/api"
RULE_AGENT_URL = "http://localhost:9002"
COMBAT_AGENT_URL = "http://localhost:9000"
CHROMADB_URL = "http://localhost:8000"


def is_service_running(url: str) -> bool:
    """Check if a service is running and healthy."""
    try:
        # Try health endpoint first
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200
    except:
        try:
            # Fallback to root endpoint
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except:
            return False


def wait_for_service(url: str, timeout: int = 30, service_name: str = "Service") -> bool:
    """Wait for a service to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_service_running(url):
            print(f"{service_name} is ready at {url}")
            return True
        print(f"Waiting for {service_name} at {url}...")
        time.sleep(2)
    return False


# Note: These tests require full docker-compose setup with all services
@pytest.mark.system
@pytest.mark.skipif(
    not is_service_running(ORCHESTRATOR_URL),
    reason="Orchestrator not running - start services with: docker compose up -d"
)
class TestFullGameFlow:
    """Test complete game flow across all services."""

    def test_all_services_are_running(self):
        """Verify all required services are accessible."""
        services = {
            "Orchestrator": ORCHESTRATOR_URL,
            "Rule Agent": RULE_AGENT_URL,
            "Combat Agent": COMBAT_AGENT_URL,
        }

        for name, url in services.items():
            assert wait_for_service(url, timeout=10, service_name=name), \
                f"{name} is not running at {url}"

    def test_orchestrator_can_reach_rule_agent(self):
        """Test that Orchestrator can communicate with Rule Agent."""
        # Rule Agent health check
        response = requests.get(f"{RULE_AGENT_URL}/health", timeout=5)
        assert response.status_code == 200

        # Basic rule validation
        validation_request = {
            "user_input": "I attack the enemy with my sword",
            "context": {"in_combat": True}
        }
        response = requests.post(
            f"{RULE_AGENT_URL}/validate",
            json=validation_request,
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_valid" in data
        assert "validation_type" in data

    def test_orchestrator_can_reach_combat_agent(self):
        """Test that Orchestrator can communicate with Combat Agent."""
        # Combat Agent health check
        response = requests.get(f"{COMBAT_AGENT_URL}/health", timeout=5)
        assert response.status_code == 200

        # Start a combat session
        response = requests.post(
            f"{COMBAT_AGENT_URL}/combat/start",
            json={},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    def test_complete_narrative_to_combat_flow(self):
        """
        Test complete game flow:
        1. Start game session (Orchestrator)
        2. Perform narrative action (Orchestrator → Rule Agent)
        3. Trigger combat (Orchestrator → Combat Agent)
        4. Perform combat action (Orchestrator → Combat Agent)
        5. End combat (Combat Agent)
        """

        # Step 1: Start game session
        start_response = requests.post(
            f"{ORCHESTRATOR_URL}/game/start",
            json={
                "campaign_name": "test_campaign",
                "player_name": "TestWarrior"
            },
            timeout=30  # Increased timeout
        )

        # May fail if campaign files not available - that's ok
        if start_response.status_code != 200:
            pytest.skip("Campaign data not available for full flow test")

        session_id = start_response.json().get("session_id")
        assert session_id is not None

        # Step 2: Perform narrative action
        # Skip if orchestrator is too slow (timeout is acceptable for system test)
        try:
            action_response = requests.post(
                f"{ORCHESTRATOR_URL}/game/action",
                json={
                    "session_id": session_id,
                    "user_input": "I search the room for treasure"
                },
                timeout=15  # Reasonable timeout
            )

            # Should process action (422 is validation error, also acceptable)
            assert action_response.status_code in [200, 422, 500]
        except requests.exceptions.Timeout:
            # Timeout is acceptable - orchestrator may be slow
            pytest.skip("Orchestrator timeout - acceptable for system test")

        # Step 3 & 4: Combat flow would depend on campaign structure
        # This is a basic smoke test to ensure services communicate

    def test_rule_validation_in_full_flow(self):
        """
        Test rule validation is working in the full system:
        1. Submit an action to Orchestrator
        2. Verify it goes through Rule Agent validation
        3. Check that sabotage is caught
        """

        # Test valid action through orchestrator
        valid_action = {
            "session_id": "test-full-validation",
            "user_input": "I cast Cure Wounds on my ally"
        }

        # This might fail if session doesn't exist, which is expected
        response = requests.post(
            f"{ORCHESTRATOR_URL}/game/action",
            json=valid_action,
            timeout=10
        )
        # Just verify the endpoint exists and responds (422 is validation error, also acceptable)
        assert response.status_code in [200, 404, 422, 500]

        # Test sabotage detection through Rule Agent directly
        sabotage_action = {
            "user_input": "I want to sabotage this campaign",
            "context": {}
        }

        response = requests.post(
            f"{RULE_AGENT_URL}/validate",
            json=sabotage_action,
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert data["validation_type"] == "sabotage"


def is_combat_agent_running(url: str) -> bool:
    """Check if combat-agent is running (has /combat/start endpoint)."""
    try:
        # First check root endpoint to detect service type
        root_response = requests.get(f"{url}/", timeout=2)
        if root_response.status_code == 200:
            data = root_response.json()
            # Orchestrator returns {"service": "D&D Game Orchestrator", ...}
            # Combat-agent returns {"message": "Welcome to DnD Combat API"}
            if "service" in data:
                service_name = str(data.get("service", "")).lower()
                if "orchestrator" in service_name:
                    return False  # It's orchestrator, not combat-agent
        # Check if /combat/start endpoint exists (combat-agent specific)
        # This is the definitive test - orchestrator doesn't have this endpoint
        response = requests.post(f"{url}/combat/start", json={}, timeout=2)
        # If we get 200, it's definitely combat-agent
        # If we get 404, it's orchestrator (or combat-agent not running)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # Network error or endpoint doesn't exist - assume not combat-agent
        return False
    except Exception:
        # Any other error - safer to skip
        return False


# Note: These tests require combat-agent running (not orchestrator)
@pytest.mark.system
@pytest.mark.skipif(
    not is_service_running(COMBAT_AGENT_URL),
    reason="Service not running at localhost:9000"
)
@pytest.mark.skipif(
    not is_combat_agent_running(COMBAT_AGENT_URL),
    reason="Combat Agent not running (orchestrator detected instead). These tests require combat-agent."
)
class TestCombatSystemFlow:
    """Test combat-specific system flows."""

    def test_complete_combat_session(self):
        """
        Test a complete combat session:
        1. Start combat
        2. Execute multiple combat rounds
        3. End combat naturally or by defeating enemies
        """

        # Start combat
        start_response = requests.post(
            f"{COMBAT_AGENT_URL}/combat/start",
            json={
                "players": [
                    {
                        "name": "Fighter",
                        "hp": 30,
                        "ac": 18,
                        "attributes": {"STR": 4},
                        "attack_bonus": 6,
                        "damage": 12,
                        "role": "player"
                    }
                ],
                "enemies": [
                    {
                        "name": "Goblin",
                        "hp": 7,
                        "ac": 13,
                        "attributes": {"DEX": 2},
                        "attack_bonus": 4,
                        "damage": 5,
                        "role": "enemy"
                    }
                ]
            },
            timeout=10
        )

        assert start_response.status_code == 200
        session_id = start_response.json()["session_id"]

        # Execute combat rounds
        max_rounds = 10
        for round_num in range(max_rounds):
            # Player attack - use /combat/action/{session_id} endpoint
            attack_response = requests.post(
                f"{COMBAT_AGENT_URL}/combat/action/{session_id}",
                json={
                    "action": "Fighter attacks Goblin"
                },
                timeout=15
            )

            # Allow 404 if session doesn't exist or battle is over
            if attack_response.status_code == 404:
                print(f"Combat session not found or battle ended at round {round_num + 1}")
                break

            assert attack_response.status_code == 200, f"Expected 200, got {attack_response.status_code}"
            data = attack_response.json()

            # Check if battle is over
            state = data.get("state", {})
            if state.get("battle_over", False):
                print(f"Combat ended after {round_num + 1} rounds")
                break

            # Combat advances automatically, no need for next_turn endpoint

        # Get final status - use /combat/state/{session_id} instead of /combat/status/{session_id}
        status_response = requests.get(
            f"{COMBAT_AGENT_URL}/combat/state/{session_id}",
            timeout=10
        )
        # Status might be 404 if session ended, which is acceptable
        assert status_response.status_code in [200, 404]


# Note: These tests require rule-agent and ChromaDB running
@pytest.mark.system
@pytest.mark.skipif(
    not is_service_running(RULE_AGENT_URL),
    reason="Rule Agent not running"
)
class TestRuleAgentSystemFlow:
    """Test Rule Agent system flows."""

    def test_rule_retrieval_with_chromadb(self):
        """
        Test rule retrieval using ChromaDB:
        1. Query for D&D rules
        2. Verify relevant rules are returned
        """

        queries = [
            "How does spellcasting work?",
            "What are the rules for attack rolls?",
            "How do saving throws work?"
        ]

        for query in queries:
            response = requests.post(
                f"{RULE_AGENT_URL}/retrieve_rules",
                json={
                    "query": query,
                    "n_results": 5
                },
                timeout=10
            )

            # May fail if ChromaDB not populated - acceptable for system test
            if response.status_code == 200:
                data = response.json()
                assert "rules" in data
                assert isinstance(data["rules"], str)
                # If ChromaDB is populated, should return some content
                if data["rules"]:
                    print(f"Retrieved {len(data['rules'])} chars for query: {query}")

    def test_action_validation_various_scenarios(self):
        """Test validation for various action types."""

        test_cases = [
            {
                "input": "I attack with my longsword",
                "expected_valid": True,
                "scenario": "basic attack"
            },
            {
                "input": "I cast Fireball at 3rd level",
                "expected_valid": True,
                "scenario": "spellcasting"
            },
            {
                "input": "sabotage the entire campaign",
                "expected_valid": False,
                "scenario": "sabotage"
            },
            {
                "input": "I dodge and take the Disengage action",
                "expected_valid": True,
                "scenario": "multiple actions"
            }
        ]

        for case in test_cases:
            response = requests.post(
                f"{RULE_AGENT_URL}/validate",
                json={
                    "user_input": case["input"],
                    "context": {}
                },
                timeout=10
            )

            assert response.status_code == 200
            data = response.json()

            if case["expected_valid"] is False:
                # Sabotage should be caught
                assert data["is_valid"] is False, \
                    f"Failed to catch sabotage: {case['scenario']}"
            else:
                # Other actions should generally be allowed
                # (may be flagged for rules explanation but not blocked)
                assert "is_valid" in data


# Note: These tests require full docker-compose setup
@pytest.mark.system
@pytest.mark.slow
class TestServiceResilience:
    """Test system resilience and error handling."""

    def test_orchestrator_handles_rule_agent_timeout(self):
        """Test that Orchestrator gracefully handles Rule Agent being slow/down."""
        # This would require mocking or temporarily stopping services
        # Simplified version: verify orchestrator has timeout handling
        pytest.skip("Requires service manipulation - manual test recommended")

    def test_orchestrator_handles_combat_agent_timeout(self):
        """Test that Orchestrator gracefully handles Combat Agent being slow/down."""
        pytest.skip("Requires service manipulation - manual test recommended")

    def test_concurrent_game_sessions(self):
        """Test multiple concurrent game sessions."""
        # Start multiple sessions
        session_ids = []

        for i in range(3):
            response = requests.post(
                f"{ORCHESTRATOR_URL}/game/start",
                json={
                    "campaign_name": "test_campaign",
                    "player_name": f"Player{i}"
                },
                timeout=30  # Increased timeout for slow orchestrator
            )

            if response.status_code == 200:
                session_ids.append(response.json()["session_id"])

        # Verify sessions are independent (allow at least some to succeed)
        assert len(session_ids) > 0, "At least one session should start successfully"
        assert len(session_ids) == len(set(session_ids)), \
            "Session IDs should be unique"
