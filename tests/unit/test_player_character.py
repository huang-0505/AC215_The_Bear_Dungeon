"""Unit tests for PlayerCharacter (src/orchestrator/models/player.py)."""

from dataclasses import FrozenInstanceError

import pytest

from models.player import PlayerCharacter, _CLASS_STATS, DEFAULT_CLASS


@pytest.mark.unit
class TestPlayerCharacterFromClass:
    def test_known_class_uses_class_stats(self):
        pc = PlayerCharacter.from_class(
            character_class="Wizard", character_name="Eldrin"
        )
        expected = _CLASS_STATS["Wizard"]
        assert pc.character_class == "Wizard"
        assert pc.character_name == "Eldrin"
        assert pc.hp == expected["hp"]
        assert pc.max_hp == expected["hp"]
        assert pc.ac == expected["ac"]
        assert pc.attack_bonus == expected["attack_bonus"]
        assert pc.attributes == expected["attributes"]

    def test_unknown_class_falls_back_to_default(self):
        pc = PlayerCharacter.from_class(
            character_class="Necromancer", character_name="Mordred"
        )
        # Stats should match the default class table
        default_stats = _CLASS_STATS[DEFAULT_CLASS]
        assert pc.hp == default_stats["hp"]
        assert pc.ac == default_stats["ac"]
        # ...but identity fields preserve the user's choice
        assert pc.character_class == "Necromancer"
        assert pc.character_name == "Mordred"

    def test_player_id_assigned_when_omitted(self):
        a = PlayerCharacter.from_class(character_class="Fighter", character_name="A")
        b = PlayerCharacter.from_class(character_class="Fighter", character_name="B")
        assert a.player_id != b.player_id

    def test_player_id_preserved_when_provided(self):
        pc = PlayerCharacter.from_class(
            character_class="Cleric",
            character_name="Lyra",
            player_id="player-123",
        )
        assert pc.player_id == "player-123"

    def test_default_name_falls_back_to_character_name(self):
        pc = PlayerCharacter.from_class(
            character_class="Rogue", character_name="Sly"
        )
        assert pc.name == "Sly"


@pytest.mark.unit
class TestPlayerCharacterSerialization:
    def test_to_dict_then_from_dict_roundtrip(self):
        original = PlayerCharacter.from_class(
            character_class="Ranger",
            character_name="Aria",
            name="Aria of the Wood",
            player_id="abc",
        )
        rebuilt = PlayerCharacter.from_dict(original.to_dict())
        assert rebuilt == original

    def test_to_combat_stats_uses_character_name(self):
        pc = PlayerCharacter.from_class(
            character_class="Barbarian", character_name="Krug"
        )
        stats = pc.to_combat_stats()
        assert stats["name"] == "Krug"
        assert stats["hp"] == pc.hp
        assert stats["player_id"] == pc.player_id


@pytest.mark.unit
class TestPlayerCharacterImmutability:
    def test_cannot_assign_attribute_directly(self):
        pc = PlayerCharacter.from_class(
            character_class="Fighter", character_name="Brick"
        )
        with pytest.raises(FrozenInstanceError):
            pc.hp = 1  # type: ignore[misc]

    def test_with_hp_returns_new_instance(self):
        pc = PlayerCharacter.from_class(
            character_class="Fighter", character_name="Brick"
        )
        damaged = pc.with_hp(5)
        assert damaged is not pc
        assert damaged.hp == 5
        assert damaged.is_alive is True
        assert pc.hp == 20  # original unchanged

    def test_with_hp_zero_marks_dead(self):
        pc = PlayerCharacter.from_class(
            character_class="Wizard", character_name="Pop"
        )
        downed = pc.with_hp(0)
        assert downed.is_alive is False

    def test_attributes_dict_cannot_be_mutated_in_place(self):
        pc = PlayerCharacter.from_class(
            character_class="Fighter", character_name="Brick"
        )
        with pytest.raises(TypeError):
            pc.attributes["STR"] = 99  # type: ignore[index]
