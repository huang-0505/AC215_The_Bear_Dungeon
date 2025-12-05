"""
Pytest configuration and shared fixtures for DnD Master tests.
This file is automatically discovered by pytest and provides fixtures to all test files.
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add source directories to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "backend-tmp"))
sys.path.insert(0, str(project_root / "src" / "orchestrator"))
sys.path.insert(0, str(project_root / "src" / "rule_agent"))


# ========== Combat API Fixtures ==========

@pytest.fixture
def api_test_client():
    """Provide FastAPI TestClient for Combat API."""
    # Clear any existing 'service' module from cache to avoid conflicts
    if 'api.service' in sys.modules:
        del sys.modules['api.service']
    if 'api.routers.combat' in sys.modules:
        del sys.modules['api.routers.combat']
    
    from api.service import app
    return TestClient(app)


# ========== Character Fixtures ==========

@pytest.fixture
def sample_player():
    """Create a sample player character for testing."""
    from api.utils.combat_engine import Character
    return Character(
        name="TestKnight",
        char_id=1,
        hp=20,
        ac=18,
        attributes={"STR": 4, "DEX": 2, "INT": 1},
        attack_bonus=6,
        damage=10,
        role="player"
    )


@pytest.fixture
def sample_enemy():
    """Create a sample enemy character for testing."""
    from api.utils.combat_engine import Character
    return Character(
        name="TestGoblin",
        char_id=2,
        hp=12,
        ac=15,
        attributes={"STR": 2, "DEX": 3, "INT": 1},
        attack_bonus=4,
        damage=6,
        role="enemy"
    )


@pytest.fixture
def sample_characters(sample_player, sample_enemy):
    """Create a set of sample characters (players and enemies) for testing."""
    from api.utils.combat_engine import Character
    
    # Create additional characters
    player2 = Character(
        name="TestRanger",
        char_id=3,
        hp=18,
        ac=16,
        attributes={"STR": 2, "DEX": 4, "INT": 2},
        attack_bonus=5,
        damage=8,
        role="player"
    )
    
    enemy2 = Character(
        name="TestOrc",
        char_id=4,
        hp=15,
        ac=13,
        attributes={"STR": 3, "DEX": 1, "INT": 1},
        attack_bonus=5,
        damage=8,
        role="enemy"
    )
    
    return {
        "players": [sample_player, player2],
        "enemies": [sample_enemy, enemy2]
    }


# ========== Combat Engine Fixtures ==========

@pytest.fixture
def combat_engine(sample_characters):
    """Create an initialized CombatEngine with sample characters."""
    from api.utils.combat_engine import CombatEngine
    
    engine = CombatEngine(
        players=sample_characters["players"],
        enemies=sample_characters["enemies"]
    )
    return engine


# ========== API Request Fixtures ==========

@pytest.fixture
def sample_combat_request():
    """Create a sample combat start request payload."""
    return {
        "players": [
            {
                "name": "TestKnight",
                "max_hp": 20,
                "ac": 18,
                "attributes": {"STR": 4, "DEX": 2, "INT": 1},
                "attack_bonus": 6,
                "damage": 10,
                "role": "player"
            }
        ],
        "enemies": [
            {
                "name": "TestGoblin",
                "max_hp": 12,
                "ac": 15,
                "attributes": {"STR": 2, "DEX": 3, "INT": 1},
                "attack_bonus": 4,
                "damage": 6,
                "role": "enemy"
            }
        ]
    }

