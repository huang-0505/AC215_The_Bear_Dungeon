"""
combat_engine.py
Rule-based combat engine for turn-based DnD-style battles.
Handles character state, battle flow, initiative, action resolution, and combat logic.
"""

import random
from collections import deque
from abc import ABC, abstractmethod
from typing import List, Optional, Dict


# ========== Character Model ==========
class Character:
    """Represents a combatant in the battle with stats, HP, and status."""

    def __init__(self, name: str, char_id: int, hp: int, ac: int, attributes: Dict[str, int],
                 attack_bonus: int, damage: int, role: str = "player"):
        self.name = name
        self.id = char_id
        self.hp = hp
        self.max_hp = hp
        self.ac = ac  # Armor Class
        self.attributes = attributes  # e.g. {"STR": 3, "DEX": 2, "INT": 5}
        self.attack_bonus = attack_bonus
        self.damage = damage
        self.role = role  # "player" or "enemy"
        self.alive = True
        self.status_effects = []

    def take_damage(self, dmg: int):
        """Apply damage to character and check if defeated."""
        self.hp -= dmg
        if self.hp <= 0:
            self.hp = 0
            self.alive = False

    def heal(self, amount: int):
        """Restore HP, capped at max HP."""
        if self.alive:
            self.hp += amount
            self.hp = min(self.hp, self.max_hp)

    def __repr__(self):
        return f"{self.name} (HP: {self.hp}/{self.max_hp}, AC: {self.ac}, Alive: {self.alive})"


# ========== Battle State Tracker ==========
class BattleState:
    """Tracks all combatants and provides query methods for battle state."""

    def __init__(self, players: List[Character], enemies: List[Character]):
        self.players = players
        self.enemies = enemies

    def get_all(self) -> List[Character]:
        """Return all combatants regardless of status."""
        return self.players + self.enemies

    def get_alive(self, role: Optional[str] = None) -> List[Character]:
        """Return alive combatants, optionally filtered by role."""
        if role == "player":
            return [p for p in self.players if p.alive]
        elif role == "enemy":
            return [e for e in self.enemies if e.alive]
        return [c for c in self.get_all() if c.alive]

    def get_by_name(self, name: str) -> Optional[Character]:
        """Find character by name (case-insensitive)."""
        for c in self.get_all():
            if c.name.lower() == name.lower() and c.alive:
                return c
        return None

    def get_by_id(self, char_id: int, role: Optional[str] = None) -> Optional[Character]:
        """Find character by ID, optionally filtered by role."""
        if role == "player":
            all_chars = self.players
        elif role == "enemy":
            all_chars = self.enemies
        else:
            all_chars = self.get_all()

        for char in all_chars:
            if char.id == char_id and char.alive:
                return char
        return None


# ========== Abstract Action Base ==========
class Action(ABC):
    """Base class for all combat actions."""

    def __init__(self, actor: Character, target: Optional[Character] = None):
        self.actor = actor
        self.target = target

    @abstractmethod
    def execute(self, engine: 'CombatEngine') -> str:
        """Execute the action and return a descriptive message."""
        pass


# ========== Concrete Action Classes ==========
class MeleeAttack(Action):
    """Close-range physical attack."""

    def execute(self, engine: 'CombatEngine') -> str:
        roll = random.randint(1, 20) + self.actor.attack_bonus
        if roll >= self.target.ac:
            dmg = random.randint(1, self.actor.damage) + self.actor.attributes.get("STR", 0)
            self.target.take_damage(dmg)
            return f"⚔️ {self.actor.name} slashes {self.target.name} for {dmg} damage!"
        else:
            return f"❌ {self.actor.name} swings and misses {self.target.name}."


class RangedAttack(Action):
    """Long-range attack with bow, crossbow, or thrown weapon."""

    def execute(self, engine: 'CombatEngine') -> str:
        roll = random.randint(1, 20) + self.actor.attributes.get("DEX", 0)
        if roll >= self.target.ac:
            dmg = random.randint(1, self.actor.damage)
            self.target.take_damage(dmg)
            return f"🏹 {self.actor.name} shoots {self.target.name} for {dmg} damage!"
        else:
            return f"❌ {self.actor.name}'s shot misses {self.target.name}."


class SpellAttack(Action):
    """Magical attack requiring a saving throw."""

    def __init__(self, actor: Character, target: Character, spell_name: str = "Fireball"):
        super().__init__(actor, target)
        self.spell_name = spell_name

    def execute(self, engine: 'CombatEngine') -> str:
        spell_dc = 10 + self.actor.attributes.get("INT", 0)
        save_roll = random.randint(1, 20) + self.target.attributes.get("WIS", 0)
        if save_roll < spell_dc:
            dmg = random.randint(1, self.actor.damage) + self.actor.attributes.get("INT", 0)
            self.target.take_damage(dmg)
            return f"🔥 {self.actor.name}'s {self.spell_name} scorches {self.target.name} for {dmg} damage!"
        else:
            return f"🛡️ {self.target.name} resists {self.actor.name}'s {self.spell_name}!"


class Heal(Action):
    """Restoration action that recovers HP."""

    def execute(self, engine: 'CombatEngine') -> str:
        heal_amount = random.randint(5, 10) + self.actor.attributes.get("WIS", 0)
        self.target.heal(heal_amount)
        return f"✨ {self.actor.name} heals {self.target.name} for {heal_amount} HP."


class Flee(Action):
    """Attempt to escape from battle."""

    def execute(self, engine: 'CombatEngine') -> str:
        chance = random.random()
        if chance > 0.5:
            engine.remove_combatant(self.actor)
            return f"🏃 {self.actor.name} successfully flees from battle!"
        else:
            return f"❌ {self.actor.name} tries to flee but is blocked!"


# ========== Action Registry ==========
ACTION_REGISTRY: Dict[int, type] = {
    0: MeleeAttack,
    1: SpellAttack,
    2: RangedAttack,
    3: Heal,
    4: Flee,
}


# ========== Action Dispatcher ==========
class ActionDispatcher:
    """Resolves actions by mapping action IDs to Action classes."""

    def __init__(self, engine: 'CombatEngine'):
        self.engine = engine

    def resolve_action(self, actor: Character, action_data: Dict) -> str:
        """
        Execute an action based on action_data containing:
        - id: action type ID
        - target: target Character object
        """
        action_id = action_data["id"]
        target = action_data["target"]

        action_cls = ACTION_REGISTRY.get(action_id)
        if not action_cls:
            raise ValueError(f"Unknown action ID: {action_id}")

        action = action_cls(actor, target)
        return action.execute(self.engine)


# ========== Combat Engine ==========
class CombatEngine:
    """Main combat orchestrator managing turns, initiative, and battle flow."""

    def __init__(self, players: List[Character], enemies: List[Character]):
        self.state = BattleState(players, enemies)
        self.dispatcher = ActionDispatcher(self)
        self.round = 1
        self.turn_queue = self.roll_initiative()
        self.current_actor: Optional[Character] = None

    def roll_initiative(self) -> deque:
        """Roll initiative for all combatants and return turn order.
        Players always go first as a workaround for enemy turn freezing issues.
        """
        chars = self.state.get_all()
        
        # Separate players and enemies
        players = [c for c in chars if c.role == "player"]
        enemies = [c for c in chars if c.role == "enemy"]
        
        # Roll initiative within each group
        player_scores = [
            (c, random.randint(1, 20) + c.attributes.get("DEX", 0)) for c in players
        ]
        enemy_scores = [
            (c, random.randint(1, 20) + c.attributes.get("DEX", 0)) for c in enemies
        ]
        
        # Sort each group by initiative
        ordered_players = sorted(player_scores, key=lambda x: x[1], reverse=True)
        ordered_enemies = sorted(enemy_scores, key=lambda x: x[1], reverse=True)
        
        # Players always go first, then enemies
        turn_order = [c for c, _ in ordered_players] + [c for c, _ in ordered_enemies]
        return deque(turn_order)

    def next_turn(self) -> Character:
        """Advance to the next combatant's turn."""
        if not self.turn_queue:
            self.round += 1
            self.turn_queue = self.roll_initiative()

        self.current_actor = self.turn_queue.popleft()
        return self.current_actor

    def process_action(self, action_data: Dict):
        """Process and execute an action."""
        result = self.dispatcher.resolve_action(self.current_actor, action_data)
        return result

    def remove_combatant(self, combatant: Character):
        """Remove a combatant from the battle (e.g., fled or defeated)."""
        if combatant in self.turn_queue:
            self.turn_queue.remove(combatant)
        combatant.alive = False

    def is_battle_over(self) -> bool:
        """Check if battle has ended (one side eliminated)."""
        players_alive = any(p.alive for p in self.state.players)
        enemies_alive = any(e.alive for e in self.state.enemies)
        return not (players_alive and enemies_alive)
