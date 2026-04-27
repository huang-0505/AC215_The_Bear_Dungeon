"""
player.py

PlayerCharacter — runtime model for a single party member.

Stored in graph state as a plain dict (Postgres checkpointer friendly);
this dataclass is the in-memory ergonomic wrapper used by API handlers and
node helpers. Stats follow the same default table as combat_node so a single
character class produces consistent stats across narration and combat.
"""

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4


# Default stats per class. Mirrors combat_node._get_player_stats so combat and
# narration agree on a player's baseline HP/AC/attack profile.
_CLASS_STATS: Mapping[str, dict] = {
    "Fighter":   {"hp": 20, "ac": 18, "attack_bonus": 7, "damage": 15, "attributes": {"STR": 4, "DEX": 2, "INT": 1}},
    "Wizard":    {"hp": 14, "ac": 12, "attack_bonus": 3, "damage": 12, "attributes": {"STR": 1, "DEX": 2, "INT": 5}},
    "Ranger":    {"hp": 16, "ac": 15, "attack_bonus": 6, "damage": 10, "attributes": {"STR": 3, "DEX": 4, "INT": 2}},
    "Cleric":    {"hp": 15, "ac": 16, "attack_bonus": 4, "damage": 8,  "attributes": {"STR": 2, "DEX": 2, "INT": 4}},
    "Barbarian": {"hp": 26, "ac": 14, "attack_bonus": 8, "damage": 18, "attributes": {"STR": 5, "DEX": 3, "INT": 1}},
    "Rogue":    {"hp": 14, "ac": 15, "attack_bonus": 5, "damage": 9,  "attributes": {"STR": 2, "DEX": 5, "INT": 3}},
}

DEFAULT_CLASS = "Fighter"


@dataclass(frozen=True)
class PlayerCharacter:
    """A single party member. Frozen — use `replace()` to mutate.

    `attributes` is wrapped with `MappingProxyType` in __post_init__ so
    callers cannot mutate the dict in-place (which `frozen=True` alone does
    not protect against).
    """

    player_id: str
    name: str
    character_class: str
    character_name: str
    hp: int
    max_hp: int
    ac: int
    attack_bonus: int
    damage: int
    attributes: Mapping[str, int] = field(default_factory=dict)
    is_alive: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.attributes, MappingProxyType):
            # frozen=True forbids regular assignment — go through object.__setattr__.
            object.__setattr__(
                self, "attributes", MappingProxyType(dict(self.attributes))
            )

    @classmethod
    def from_class(
        cls,
        *,
        character_class: str,
        character_name: str,
        name: str | None = None,
        player_id: str | None = None,
    ) -> "PlayerCharacter":
        stats = _CLASS_STATS.get(character_class, _CLASS_STATS[DEFAULT_CLASS])
        return cls(
            player_id=player_id or str(uuid4()),
            name=name or character_name,
            character_class=character_class,
            character_name=character_name,
            hp=stats["hp"],
            max_hp=stats["hp"],
            ac=stats["ac"],
            attack_bonus=stats["attack_bonus"],
            damage=stats["damage"],
            attributes=dict(stats["attributes"]),
        )

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "character_class": self.character_class,
            "character_name": self.character_name,
            "hp": self.hp,
            "max_hp": self.max_hp,
            "ac": self.ac,
            "attack_bonus": self.attack_bonus,
            "damage": self.damage,
            "attributes": dict(self.attributes),
            "is_alive": self.is_alive,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlayerCharacter":
        return cls(
            player_id=data["player_id"],
            name=data["name"],
            character_class=data["character_class"],
            character_name=data["character_name"],
            hp=data["hp"],
            max_hp=data["max_hp"],
            ac=data["ac"],
            attack_bonus=data["attack_bonus"],
            damage=data["damage"],
            attributes=dict(data.get("attributes", {})),
            is_alive=data.get("is_alive", True),
        )

    def with_hp(self, hp: int) -> "PlayerCharacter":
        return replace(self, hp=hp, is_alive=hp > 0)

    def to_combat_stats(self) -> dict:
        """Shape expected by the Combat Agent /combat/start endpoint."""
        return {
            "name": self.character_name or self.character_class,
            "hp": self.hp,
            "ac": self.ac,
            "attributes": dict(self.attributes),
            "attack_bonus": self.attack_bonus,
            "damage": self.damage,
            "player_id": self.player_id,
        }
