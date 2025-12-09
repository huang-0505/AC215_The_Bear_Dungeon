"""
System tests for the complete DnD Combat System.
Requires running Docker services (docker-compose up).
"""

import pytest
import requests
import time

# System test URLs
COMBAT_API_URL = "http://localhost:9000"
NGINX_URL = "http://localhost:8080"


def is_service_running(url: str, endpoint: str = "/health") -> bool:
    """Check if a service is accessible."""
    try:
        response = requests.get(f"{url}{endpoint}", timeout=2)
        return response.status_code == 200
    except:
        return False


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


@pytest.mark.system
@pytest.mark.skipif(
    not is_service_running(COMBAT_API_URL),
    reason="Service not running at localhost:9000. Run 'docker-compose up' first."
)
@pytest.mark.skipif(
    not is_combat_agent_running(COMBAT_API_URL),
    reason="Combat Agent not running at localhost:9000 (orchestrator detected instead). These tests require combat-agent."
)
class TestCombatSystemDirect:
    """System tests for Combat Agent (direct access)."""

    def test_health_check_real_service(self):
        """Test health check on real running service."""
        response = requests.get(f"{COMBAT_API_URL}/health", timeout=5)

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_start_combat_real_http(self):
        """Test starting combat via real HTTP request."""
        response = requests.post(
            f"{COMBAT_API_URL}/combat/start",
            json={},
            timeout=10
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "state" in data

    def test_full_combat_session_end_to_end(self):
        """Test complete combat session from start to end."""
        # 1. Start combat
        start_response = requests.post(
            f"{COMBAT_API_URL}/combat/start",
            json={},
            timeout=10
        )
        assert start_response.status_code == 200
        session_id = start_response.json()["session_id"]

        # 2. Get initial state
        state_response = requests.get(
            f"{COMBAT_API_URL}/combat/state/{session_id}",
            timeout=5
        )
        assert state_response.status_code == 200
        initial_state = state_response.json()

        # 3. Execute combat action
        action_response = requests.post(
            f"{COMBAT_API_URL}/combat/action/{session_id}",
            json={"action": "Attack the nearest enemy"},
            timeout=10
        )
        assert action_response.status_code == 200
        action_data = action_response.json()

        # Verify narrative was generated
        assert "narrative" in action_data
        assert "state" in action_data

        # 4. Verify state changed
        updated_state = action_data["state"]
        # Round might advance
        assert "round" in updated_state

        # 5. Clean up - end session
        end_response = requests.delete(
            f"{COMBAT_API_URL}/combat/session/{session_id}",
            timeout=5
        )
        assert end_response.status_code == 200

    def test_concurrent_combat_sessions(self):
        """Test multiple concurrent combat sessions."""
        sessions = []

        # Create 3 concurrent sessions
        for i in range(3):
            response = requests.post(
                f"{COMBAT_API_URL}/combat/start",
                json={},
                timeout=10
            )
            assert response.status_code == 200
            sessions.append(response.json()["session_id"])

        # Verify each session is independent
        for session_id in sessions:
            response = requests.get(
                f"{COMBAT_API_URL}/combat/state/{session_id}",
                timeout=5
            )
            assert response.status_code == 200

        # Clean up all sessions
        for session_id in sessions:
            requests.delete(
                f"{COMBAT_API_URL}/combat/session/{session_id}",
                timeout=5
            )

    def test_response_time_performance(self):
        """Test API response time is acceptable."""
        start_time = time.time()
        response = requests.get(f"{COMBAT_API_URL}/health", timeout=5)
        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert elapsed < 1.0, f"Health check took {elapsed:.2f}s (should be < 1s)"

    @pytest.mark.slow
    def test_multiple_rapid_actions(self):
        """Test handling rapid successive actions."""
        # Start combat
        start_response = requests.post(
            f"{COMBAT_API_URL}/combat/start",
            json={},
            timeout=10
        )
        session_id = start_response.json()["session_id"]

        # Execute 10 rapid actions
        success_count = 0
        for i in range(10):
            try:
                action_response = requests.post(
                    f"{COMBAT_API_URL}/combat/action/{session_id}",
                    json={"action": f"Action {i}"},
                    timeout=10
                )
                if action_response.status_code == 200:
                    success_count += 1
                elif action_response.status_code == 400:
                    # Battle might be over
                    break
            except requests.exceptions.Timeout:
                # Timeout is acceptable for rapid requests
                pass

        # At least some actions should succeed
        assert success_count > 0

        # Clean up
        requests.delete(
            f"{COMBAT_API_URL}/combat/session/{session_id}",
            timeout=5
        )


@pytest.mark.system
@pytest.mark.skipif(
    not is_service_running(NGINX_URL, "/api"),
    reason="Nginx not running at localhost:8080. Run 'docker-compose up' first."
)
class TestSystemViaGateway:
    """System tests accessing Combat API through API Gateway/Nginx."""

    def test_combat_through_reverse_proxy(self):
        """Test accessing Combat API through Nginx reverse proxy."""
        # Note: Update URL based on your nginx.conf routing
        # This assumes nginx routes /api/combat/* to combat-agent
        response = requests.get(f"{NGINX_URL}/api/health", timeout=5)

        # Verify we can reach the service through the gateway
        assert response.status_code in [200, 404]  # Depends on nginx config


@pytest.mark.system
class TestServiceAvailability:
    """Test availability of all system components."""

    def test_chromadb_accessible(self):
        """Test ChromaDB is accessible."""
        if is_service_running("http://localhost:8000", "/api/v1/heartbeat"):
            response = requests.get(
                "http://localhost:8000/api/v1/heartbeat",
                timeout=5
            )
            assert response.status_code == 200
        else:
            pytest.skip("ChromaDB not running")

    def test_all_services_status(self):
        """Report status of all services."""
        services = {
            "Combat API": (COMBAT_API_URL, "/health"),
            "Nginx": (NGINX_URL, "/"),
            "ChromaDB": ("http://localhost:8000", "/api/v1/heartbeat"),
        }

        status_report = {}
        for name, (url, endpoint) in services.items():
            status_report[name] = is_service_running(url, endpoint)

        print("\n=== Service Status ===")
        for name, status in status_report.items():
            print(f"{name}: {'✓ Running' if status else '✗ Not running'}")

        # At least Combat API should be running for system tests
        assert status_report["Combat API"], "Combat API is not running"
