"""
rooms.py

Redis-backed multiplayer room manager.

A room is a stable container that:
- holds 1..MAX_PARTY_SIZE players,
- has one LangGraph thread (room_id == LangGraph thread_id) once started,
- collects per-round actions (one per player) before invoking the graph,
- broadcasts state updates to subscribers via Redis pub/sub.

Redis key layout
----------------
room:{room_id}                           Hash:  room metadata + game state pointer
room:{room_id}:players                   Hash:  player_id -> JSON PlayerCharacter
room:{room_id}:actions:{round_no}        Hash:  player_id -> action text (current round)
room:{room_id}:lock                      String: SETNX lock for graph invocation
room:{room_id}:events                    Pub/sub channel for SSE fanout
rooms:index                              Hash:  room_id -> created_at (lobby listing)
"""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from redis_client import get_redis
from models.player import PlayerCharacter

logger = logging.getLogger(__name__)

MAX_PARTY_SIZE = 4
ROOM_TTL_SECONDS = 60 * 60 * 24  # auto-expire abandoned rooms after 24h
LOCK_TTL_SECONDS = 30
ROOM_INDEX_KEY = "rooms:index"

# Room state values
ROOM_STATE_LOBBY = "lobby"        # players gathering, host hasn't started
ROOM_STATE_ACTIVE = "active"      # game running
ROOM_STATE_ENDED = "ended"


def _room_key(room_id: str) -> str:
    return f"room:{room_id}"


def _players_key(room_id: str) -> str:
    return f"room:{room_id}:players"


def _actions_key(room_id: str, round_no: int) -> str:
    return f"room:{room_id}:actions:{round_no}"


def _lock_key(room_id: str) -> str:
    return f"room:{room_id}:lock"


def _tokens_key(room_id: str) -> str:
    return f"room:{room_id}:tokens"


def events_channel(room_id: str) -> str:
    return f"room:{room_id}:events"


@dataclass(frozen=True)
class Room:
    """In-memory snapshot of a room. Read via RoomManager.get_room()."""

    room_id: str
    name: str
    state: str
    host_player_id: Optional[str]
    campaign_id: Optional[str]
    max_players: int
    created_at: int
    current_round: int

    def to_dict(self) -> dict:
        return {
            "room_id": self.room_id,
            "name": self.name,
            "state": self.state,
            "host_player_id": self.host_player_id,
            "campaign_id": self.campaign_id,
            "max_players": self.max_players,
            "created_at": self.created_at,
            "current_round": self.current_round,
        }


class RoomError(Exception):
    """Domain error: invalid room operation (full, missing player, etc.)."""


class RoomManager:
    """All room operations. Stateless — Redis is the source of truth."""

    @staticmethod
    def create_room(name: str, host_player_id: str, max_players: int = MAX_PARTY_SIZE) -> Room:
        if max_players > MAX_PARTY_SIZE:
            raise RoomError(f"max_players cannot exceed {MAX_PARTY_SIZE}")

        room_id = str(uuid4())
        now = int(time.time())
        meta = {
            "room_id": room_id,
            "name": name,
            "state": ROOM_STATE_LOBBY,
            "host_player_id": host_player_id,
            "campaign_id": "",
            "max_players": str(max_players),
            "created_at": str(now),
            "current_round": "0",
        }
        client = get_redis()
        pipe = client.pipeline()
        pipe.hset(_room_key(room_id), mapping=meta)
        pipe.expire(_room_key(room_id), ROOM_TTL_SECONDS)
        pipe.hset(ROOM_INDEX_KEY, room_id, now)
        pipe.execute()

        logger.info("Created room %s host=%s", room_id, host_player_id)
        return RoomManager._meta_to_room(meta)

    @staticmethod
    def get_room(room_id: str) -> Optional[Room]:
        meta = get_redis().hgetall(_room_key(room_id))
        if not meta:
            return None
        return RoomManager._meta_to_room(meta)

    @staticmethod
    def list_rooms() -> list[Room]:
        client = get_redis()
        ids = client.hkeys(ROOM_INDEX_KEY) or []
        rooms: list[Room] = []
        for room_id in ids:
            room = RoomManager.get_room(room_id)
            if room is None:
                client.hdel(ROOM_INDEX_KEY, room_id)  # tombstone cleanup
                continue
            rooms.append(room)
        return rooms

    @staticmethod
    def add_player(room_id: str, player: PlayerCharacter) -> Room:
        client = get_redis()
        room = RoomManager.get_room(room_id)
        if room is None:
            raise RoomError("room not found")
        if room.state != ROOM_STATE_LOBBY:
            raise RoomError("cannot join: room is not in lobby state")
        current_size = client.hlen(_players_key(room_id))
        if current_size >= room.max_players:
            raise RoomError("room is full")
        if client.hexists(_players_key(room_id), player.player_id):
            raise RoomError("player already in room")

        # Combat agent identifies actors by character_name; collisions inside a
        # single party would make turn routing ambiguous. Enforce uniqueness.
        existing = RoomManager.list_players(room_id)
        if any(p.character_name == player.character_name for p in existing):
            raise RoomError(
                f"character_name '{player.character_name}' is already taken in this room"
            )

        client.hset(_players_key(room_id), player.player_id, json.dumps(player.to_dict()))
        client.expire(_players_key(room_id), ROOM_TTL_SECONDS)

        RoomManager._publish(room_id, {"type": "player-joined", "player_id": player.player_id})
        return room

    @staticmethod
    def remove_player(room_id: str, player_id: str) -> Optional[Room]:
        client = get_redis()
        room = RoomManager.get_room(room_id)
        if room is None:
            return None
        client.hdel(_players_key(room_id), player_id)
        RoomManager.revoke_token(room_id, player_id)

        remaining = client.hlen(_players_key(room_id))
        if remaining == 0:
            RoomManager._delete_room(room_id)
            return None

        # If host left, hand the role to any remaining player.
        if room.host_player_id == player_id:
            survivors = client.hkeys(_players_key(room_id))
            new_host = survivors[0] if survivors else None
            if new_host:
                client.hset(_room_key(room_id), "host_player_id", new_host)
                room = RoomManager.get_room(room_id)

        RoomManager._publish(room_id, {"type": "player-left", "player_id": player_id})
        return room

    @staticmethod
    def list_players(room_id: str) -> list[PlayerCharacter]:
        raw = get_redis().hvals(_players_key(room_id)) or []
        return [PlayerCharacter.from_dict(json.loads(r)) for r in raw]

    @staticmethod
    def mark_active(room_id: str, campaign_id: Optional[str]) -> None:
        get_redis().hset(
            _room_key(room_id),
            mapping={"state": ROOM_STATE_ACTIVE, "campaign_id": campaign_id or ""},
        )
        RoomManager._publish(room_id, {"type": "room-started", "campaign_id": campaign_id})

    @staticmethod
    def mark_ended(room_id: str) -> None:
        get_redis().hset(_room_key(room_id), "state", ROOM_STATE_ENDED)
        RoomManager._publish(room_id, {"type": "room-ended"})

    @staticmethod
    def submit_action(room_id: str, round_no: int, player_id: str, action: str) -> dict[str, str]:
        """Atomically record a player's action for the current round.
        Returns the full pending_actions hash after the write."""
        client = get_redis()
        if not client.hexists(_players_key(room_id), player_id):
            raise RoomError("player is not in this room")

        key = _actions_key(room_id, round_no)
        client.hset(key, player_id, action)
        client.expire(key, ROOM_TTL_SECONDS)

        actions = client.hgetall(key) or {}
        RoomManager._publish(room_id, {
            "type": "action-submitted",
            "player_id": player_id,
            "round": round_no,
            "submitted": list(actions.keys()),
        })
        return actions

    @staticmethod
    def clear_round_actions(room_id: str, round_no: int) -> None:
        get_redis().delete(_actions_key(room_id, round_no))

    @staticmethod
    def increment_round(room_id: str) -> int:
        new_round = int(get_redis().hincrby(_room_key(room_id), "current_round", 1))
        return new_round

    @staticmethod
    def acquire_invoke_lock(room_id: str) -> bool:
        """Try to acquire the per-room graph-invocation lock. Returns True if held."""
        return bool(
            get_redis().set(_lock_key(room_id), "1", nx=True, ex=LOCK_TTL_SECONDS)
        )

    @staticmethod
    def release_invoke_lock(room_id: str) -> None:
        get_redis().delete(_lock_key(room_id))

    # ---- player tokens (lightweight no-account auth) ----
    #
    # Each player is issued a high-entropy opaque token at room create/join.
    # The token is returned ONCE in the HTTP response and never broadcast in
    # SSE events. Subsequent room mutations must present the matching token
    # via the X-Player-Token header. Token check uses hmac.compare_digest to
    # avoid timing-side-channel leaks.

    @staticmethod
    def issue_token(room_id: str, player_id: str) -> str:
        """Generate, store, and return a fresh token for `player_id`."""
        token = secrets.token_urlsafe(32)
        client = get_redis()
        client.hset(_tokens_key(room_id), player_id, token)
        client.expire(_tokens_key(room_id), ROOM_TTL_SECONDS)
        return token

    @staticmethod
    def verify_token(room_id: str, player_id: str, token: Optional[str]) -> bool:
        """Constant-time check that the player presented the right token."""
        if not token:
            return False
        stored = get_redis().hget(_tokens_key(room_id), player_id)
        if not stored:
            return False
        return hmac.compare_digest(str(stored), str(token))

    @staticmethod
    def revoke_token(room_id: str, player_id: str) -> None:
        get_redis().hdel(_tokens_key(room_id), player_id)

    # ---- internal ----

    @staticmethod
    def _meta_to_room(meta: dict) -> Room:
        return Room(
            room_id=meta["room_id"],
            name=meta.get("name", "Untitled"),
            state=meta.get("state", ROOM_STATE_LOBBY),
            host_player_id=meta.get("host_player_id") or None,
            campaign_id=meta.get("campaign_id") or None,
            max_players=int(meta.get("max_players", MAX_PARTY_SIZE)),
            created_at=int(meta.get("created_at", 0)),
            current_round=int(meta.get("current_round", 0)),
        )

    @staticmethod
    def _publish(room_id: str, event: dict) -> None:
        try:
            get_redis().publish(events_channel(room_id), json.dumps(event))
        except Exception as e:  # noqa: BLE001 - pub/sub is best-effort
            logger.warning("Failed to publish event for room %s: %s", room_id, e)

    @staticmethod
    def _delete_room(room_id: str) -> None:
        client = get_redis()
        pipe = client.pipeline()
        pipe.delete(_room_key(room_id))
        pipe.delete(_players_key(room_id))
        pipe.delete(_tokens_key(room_id))
        pipe.hdel(ROOM_INDEX_KEY, room_id)
        pipe.execute()
        RoomManager._publish(room_id, {"type": "room-deleted"})
