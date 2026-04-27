"""
party_actions.py

Helpers for combining multiple players' submitted actions into a single
"party action" string the narrator/rule-validator/router consume.

Single-player: returns the lone action verbatim so all downstream behavior
matches the legacy single-character pipeline.
"""

from models.player import PlayerCharacter


def synthesize_party_action(
    players: list[dict], pending_actions: dict[str, str]
) -> str:
    """
    Combine each player's submitted action into one prompt.

    Args:
        players: list of PlayerCharacter dicts (the party)
        pending_actions: player_id -> raw action text

    Returns:
        Empty string if nothing submitted, the lone action for solo play, or
        a labeled multi-line block for parties of two or more.
    """
    if not pending_actions:
        return ""

    if len(pending_actions) == 1:
        return next(iter(pending_actions.values()))

    by_id: dict[str, PlayerCharacter] = {
        p["player_id"]: PlayerCharacter.from_dict(p) for p in players
    }
    lines: list[str] = ["The party acts together:"]
    for player_id, action in pending_actions.items():
        actor = by_id.get(player_id)
        if actor is None:
            label = "Adventurer"
        else:
            label = f"{actor.character_name} the {actor.character_class}"
        lines.append(f"- {label}: {action}")

    return "\n".join(lines)
