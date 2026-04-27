"""Unit tests for synthesize_party_action (src/orchestrator/party_actions.py)."""

import pytest

from models.player import PlayerCharacter
from party_actions import synthesize_party_action


def _player_dict(class_name: str, char_name: str, player_id: str) -> dict:
    return PlayerCharacter.from_class(
        character_class=class_name,
        character_name=char_name,
        player_id=player_id,
    ).to_dict()


@pytest.mark.unit
class TestSynthesizePartyAction:
    def test_no_actions_returns_empty_string(self):
        assert synthesize_party_action([], {}) == ""

    def test_single_action_returns_verbatim(self):
        players = [_player_dict("Fighter", "Brick", "p1")]
        actions = {"p1": "I charge the goblin."}
        assert (
            synthesize_party_action(players, actions) == "I charge the goblin."
        )

    def test_multiple_actions_get_labeled(self):
        players = [
            _player_dict("Wizard", "Eldrin", "p1"),
            _player_dict("Fighter", "Brick", "p2"),
        ]
        actions = {
            "p1": "I cast firebolt.",
            "p2": "I draw my axe and advance.",
        }
        result = synthesize_party_action(players, actions)
        assert result.startswith("The party acts together:")
        assert "Eldrin the Wizard: I cast firebolt." in result
        assert "Brick the Fighter: I draw my axe and advance." in result

    def test_unknown_player_id_falls_back_to_adventurer_label(self):
        players = [_player_dict("Cleric", "Lyra", "p1")]
        actions = {
            "p1": "I bless the party.",
            "ghost": "I haunt the room.",
        }
        result = synthesize_party_action(players, actions)
        assert "Lyra the Cleric: I bless the party." in result
        assert "Adventurer: I haunt the room." in result
