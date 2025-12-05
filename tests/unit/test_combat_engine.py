"""
Unit tests for combat_engine.py
Tests individual components of the combat system in isolation.
"""

import pytest
from api.utils.combat_engine import Character, BattleState, CombatEngine


@pytest.mark.unit
class TestCharacter:
    """Test Character class functionality."""

    def test_character_initialization(self, sample_player):
        """Test creating a character with proper attributes."""
        assert sample_player.name == "TestKnight"
        assert sample_player.hp == 20
        assert sample_player.max_hp == 20
        assert sample_player.ac == 18
        assert sample_player.alive is True
        assert sample_player.role == "player"

    def test_character_take_damage(self, sample_player):
        """Test damage reduces HP correctly."""
        sample_player.take_damage(5)
        assert sample_player.hp == 15
        assert sample_player.alive is True

    def test_character_death(self, sample_enemy):
        """Test character dies when HP reaches 0."""
        sample_enemy.take_damage(12)
        assert sample_enemy.hp == 0
        assert sample_enemy.alive is False

    def test_character_overkill_damage(self, sample_enemy):
        """Test damage exceeding HP stops at 0."""
        sample_enemy.take_damage(20)
        assert sample_enemy.hp == 0
        assert sample_enemy.alive is False

    def test_character_heal(self, sample_player):
        """Test healing restores HP."""
        sample_player.take_damage(10)
        sample_player.heal(5)
        assert sample_player.hp == 15

    def test_character_heal_cap_at_max_hp(self, sample_player):
        """Test healing cannot exceed max HP."""
        sample_player.take_damage(5)
        sample_player.heal(10)
        assert sample_player.hp == 20  # Should cap at max_hp

    def test_cannot_heal_dead_character(self, sample_player):
        """Test dead characters cannot be healed."""
        sample_player.take_damage(20)
        sample_player.heal(5)
        assert sample_player.hp == 0
        assert sample_player.alive is False


@pytest.mark.unit
class TestBattleState:
    """Test BattleState management."""

    def test_get_all_combatants(self, sample_characters):
        """Test retrieving all combatants."""
        state = BattleState(
            sample_characters["players"],
            sample_characters["enemies"]
        )
        all_chars = state.get_all()
        assert len(all_chars) == 4

    def test_get_alive_players(self, sample_characters):
        """Test filtering alive players."""
        state = BattleState(
            sample_characters["players"],
            sample_characters["enemies"]
        )
        alive_players = state.get_alive(role="player")
        assert len(alive_players) == 2

    def test_get_alive_enemies(self, sample_characters):
        """Test filtering alive enemies."""
        state = BattleState(
            sample_characters["players"],
            sample_characters["enemies"]
        )
        alive_enemies = state.get_alive(role="enemy")
        assert len(alive_enemies) == 2

    def test_get_alive_after_death(self, sample_characters):
        """Test alive count updates after character death."""
        players = sample_characters["players"]
        enemies = sample_characters["enemies"]
        state = BattleState(players, enemies)

        players[0].take_damage(20)
        alive_players = state.get_alive(role="player")
        assert len(alive_players) == 1

    def test_get_by_name(self, sample_characters):
        """Test finding character by name."""
        state = BattleState(
            sample_characters["players"],
            sample_characters["enemies"]
        )
        char = state.get_by_name("Knight")
        assert char is not None
        assert char.name == "Knight"

    def test_get_by_name_case_insensitive(self, sample_characters):
        """Test name search is case-insensitive."""
        state = BattleState(
            sample_characters["players"],
            sample_characters["enemies"]
        )
        char = state.get_by_name("WIZARD")
        assert char is not None
        assert char.name == "Wizard"

    def test_get_by_name_dead_character_returns_none(self, sample_characters):
        """Test dead characters are not returned by name search."""
        players = sample_characters["players"]
        enemies = sample_characters["enemies"]
        state = BattleState(players, enemies)

        players[0].take_damage(20)
        char = state.get_by_name("Knight")
        assert char is None

    def test_get_by_id(self, sample_characters):
        """Test finding character by ID."""
        state = BattleState(
            sample_characters["players"],
            sample_characters["enemies"]
        )
        char = state.get_by_id(1, role="player")
        assert char is not None
        assert char.name == "Wizard"


@pytest.mark.unit
class TestCombatEngine:
    """Test CombatEngine orchestration."""

    def test_engine_initialization(self, combat_engine):
        """Test combat engine initializes correctly."""
        assert combat_engine.round >= 0

    def test_battle_not_over_initially(self, combat_engine):
        """Test battle is not over at start."""
        assert combat_engine.is_battle_over() is False

    def test_battle_over_when_all_players_dead(self, sample_characters):
        """Test battle ends when all players are defeated."""
        players = sample_characters["players"]
        enemies = sample_characters["enemies"]
        engine = CombatEngine(players, enemies)

        for player in players:
            player.take_damage(100)

        assert engine.is_battle_over() is True

    def test_battle_over_when_all_enemies_dead(self, sample_characters):
        """Test battle ends when all enemies are defeated."""
        players = sample_characters["players"]
        enemies = sample_characters["enemies"]
        engine = CombatEngine(players, enemies)

        for enemy in enemies:
            enemy.take_damage(100)

        assert engine.is_battle_over() is True

    def test_next_turn_advances_round(self, combat_engine):
        """Test next_turn increments round counter."""
        initial_round = combat_engine.round
        combat_engine.next_turn()
        # Round should advance after going through all characters
        assert combat_engine.round >= initial_round
