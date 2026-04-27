"""Unit tests for RoomManager (src/orchestrator/rooms.py).

Uses fakeredis to avoid a real Redis connection. The orchestrator's
`redis_client.get_redis()` is monkeypatched to return the fake instance.
"""

import json

import pytest

import redis_client
from rooms import (
    MAX_PARTY_SIZE,
    ROOM_STATE_ACTIVE,
    ROOM_STATE_LOBBY,
    RoomError,
    RoomManager,
)
from models.player import PlayerCharacter


@pytest.fixture
def fake_redis(monkeypatch):
    """Replace the orchestrator's Redis client with a fakeredis instance.

    Patch both `redis_client.get_redis` (the canonical accessor) and
    `rooms.get_redis` (the name `rooms` imported at module load), since
    `from x import y` snapshots the binding at import time.
    """
    import fakeredis

    import rooms as rooms_module

    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_client, "get_redis", lambda: client)
    monkeypatch.setattr(rooms_module, "get_redis", lambda: client)

    yield client

    client.flushall()


def _make_pc(name: str, klass: str = "Fighter", player_id: str | None = None) -> PlayerCharacter:
    return PlayerCharacter.from_class(
        character_class=klass, character_name=name, player_id=player_id
    )


@pytest.mark.unit
class TestCreateRoom:
    def test_creates_room_in_lobby_state(self, fake_redis):
        host = _make_pc("Host")
        room = RoomManager.create_room("test", host.player_id)
        assert room.state == ROOM_STATE_LOBBY
        assert room.host_player_id == host.player_id
        assert room.max_players == MAX_PARTY_SIZE
        assert room.current_round == 0

    def test_create_rejects_oversized_party(self, fake_redis):
        with pytest.raises(RoomError):
            RoomManager.create_room("test", "h1", max_players=10)

    def test_listing_includes_created_room(self, fake_redis):
        room = RoomManager.create_room("alpha", "h1")
        listed = {r.room_id for r in RoomManager.list_rooms()}
        assert room.room_id in listed


@pytest.mark.unit
class TestAddPlayer:
    def test_adds_up_to_max_players(self, fake_redis):
        host = _make_pc("Host")
        room = RoomManager.create_room("test", host.player_id, max_players=2)
        RoomManager.add_player(room.room_id, host)
        RoomManager.add_player(room.room_id, _make_pc("Second"))

    def test_full_room_rejects_new_player(self, fake_redis):
        room = RoomManager.create_room("test", "h", max_players=1)
        RoomManager.add_player(room.room_id, _make_pc("Solo"))
        with pytest.raises(RoomError, match="full"):
            RoomManager.add_player(room.room_id, _make_pc("Late"))

    def test_duplicate_character_name_rejected(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        RoomManager.add_player(room.room_id, _make_pc("Eldrin"))
        with pytest.raises(RoomError, match="character_name"):
            RoomManager.add_player(room.room_id, _make_pc("Eldrin", klass="Wizard"))

    def test_duplicate_player_id_rejected(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        original = _make_pc("First")
        RoomManager.add_player(room.room_id, original)
        with pytest.raises(RoomError, match="already"):
            RoomManager.add_player(room.room_id, original)

    def test_cannot_join_active_room(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        RoomManager.add_player(room.room_id, _make_pc("First"))
        RoomManager.mark_active(room.room_id, campaign_id=None)
        with pytest.raises(RoomError, match="lobby"):
            RoomManager.add_player(room.room_id, _make_pc("Late"))


@pytest.mark.unit
class TestRemovePlayer:
    def test_removing_last_player_deletes_room(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        pc = _make_pc("Solo")
        RoomManager.add_player(room.room_id, pc)
        result = RoomManager.remove_player(room.room_id, pc.player_id)
        assert result is None
        assert RoomManager.get_room(room.room_id) is None

    def test_host_handoff_when_host_leaves(self, fake_redis):
        host = _make_pc("Host", player_id="host-id")
        other = _make_pc("Other", player_id="other-id")
        room = RoomManager.create_room("test", host.player_id)
        RoomManager.add_player(room.room_id, host)
        RoomManager.add_player(room.room_id, other)

        result = RoomManager.remove_player(room.room_id, host.player_id)
        assert result is not None
        assert result.host_player_id == other.player_id


@pytest.mark.unit
class TestActionsAndRounds:
    def test_submit_action_records_in_round_hash(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        pc = _make_pc("Solo", player_id="solo")
        RoomManager.add_player(room.room_id, pc)
        actions = RoomManager.submit_action(room.room_id, 0, pc.player_id, "go north")
        assert actions == {pc.player_id: "go north"}

    def test_submit_action_rejects_unknown_player(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        with pytest.raises(RoomError):
            RoomManager.submit_action(room.room_id, 0, "ghost", "boo")

    def test_increment_round(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        assert RoomManager.increment_round(room.room_id) == 1
        assert RoomManager.increment_round(room.room_id) == 2
        assert RoomManager.get_room(room.room_id).current_round == 2

    def test_lock_is_exclusive(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        assert RoomManager.acquire_invoke_lock(room.room_id) is True
        assert RoomManager.acquire_invoke_lock(room.room_id) is False
        RoomManager.release_invoke_lock(room.room_id)
        assert RoomManager.acquire_invoke_lock(room.room_id) is True

    def test_clear_round_actions(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        pc = _make_pc("Solo", player_id="solo")
        RoomManager.add_player(room.room_id, pc)
        RoomManager.submit_action(room.room_id, 0, pc.player_id, "go north")
        RoomManager.clear_round_actions(room.room_id, 0)
        assert fake_redis.hgetall(f"room:{room.room_id}:actions:0") == {}


@pytest.mark.unit
class TestPlayerTokens:
    def test_issue_token_returns_high_entropy_string(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        token = RoomManager.issue_token(room.room_id, "player-1")
        # token_urlsafe(32) yields ~43 chars (32 bytes base64-encoded).
        assert isinstance(token, str)
        assert len(token) >= 32

    def test_verify_accepts_correct_token(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        token = RoomManager.issue_token(room.room_id, "player-1")
        assert RoomManager.verify_token(room.room_id, "player-1", token) is True

    def test_verify_rejects_wrong_token(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        RoomManager.issue_token(room.room_id, "player-1")
        assert RoomManager.verify_token(room.room_id, "player-1", "bogus") is False

    def test_verify_rejects_missing_token(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        RoomManager.issue_token(room.room_id, "player-1")
        assert RoomManager.verify_token(room.room_id, "player-1", None) is False
        assert RoomManager.verify_token(room.room_id, "player-1", "") is False

    def test_verify_rejects_unknown_player(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        assert RoomManager.verify_token(room.room_id, "ghost", "anything") is False

    def test_revoke_token_invalidates_subsequent_calls(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        token = RoomManager.issue_token(room.room_id, "player-1")
        RoomManager.revoke_token(room.room_id, "player-1")
        assert RoomManager.verify_token(room.room_id, "player-1", token) is False

    def test_remove_player_revokes_token(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        pc = _make_pc("Solo", player_id="solo")
        RoomManager.add_player(room.room_id, pc)
        token = RoomManager.issue_token(room.room_id, pc.player_id)
        # Add another player so the room survives the leave.
        RoomManager.add_player(room.room_id, _make_pc("Other"))
        RoomManager.remove_player(room.room_id, pc.player_id)
        assert RoomManager.verify_token(room.room_id, pc.player_id, token) is False


@pytest.mark.unit
class TestStateTransitions:
    def test_mark_active_updates_state_and_campaign(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        RoomManager.mark_active(room.room_id, campaign_id="phandelver")
        refreshed = RoomManager.get_room(room.room_id)
        assert refreshed.state == ROOM_STATE_ACTIVE
        assert refreshed.campaign_id == "phandelver"

    def test_mark_ended(self, fake_redis):
        room = RoomManager.create_room("test", "h")
        RoomManager.mark_ended(room.room_id)
        assert RoomManager.get_room(room.room_id).state == "ended"
