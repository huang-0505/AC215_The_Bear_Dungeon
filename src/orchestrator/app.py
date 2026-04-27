"""
app.py - LangGraph-based D&D Game Orchestrator

Multiplayer-aware: state.players holds the party (1..4), every action is
collected per player_id. Solo play is just a 1-player party.
"""

import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from uuid import uuid4, UUID
from typing import Dict, Optional, List, AsyncIterator

import requests
import redis.asyncio as redis_async
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from rate_limit import (
    LIMIT_CREATE_ROOM,
    LIMIT_GAME_ACTION,
    LIMIT_ROOM_ACTION,
    LIMIT_SSE_CONNECT,
    limiter,
)

from graph import invoke_game_action, get_session_state, get_graph
from campaign_loader import CampaignLoader
from story_tree_loader import StoryTreeLoader
from nodes.narrator_node import generate_initial_choices
from redis_client import get_redis
from models.player import PlayerCharacter
from party_actions import synthesize_party_action
from rooms import (
    RoomManager,
    RoomError,
    Room,
    MAX_PARTY_SIZE as ROOM_MAX_PARTY_SIZE,
    ROOM_STATE_LOBBY,
    ROOM_STATE_ACTIVE,
    events_channel,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PARTY_SIZE = 4

# Single async Redis client shared by every SSE subscriber so we don't open
# a new TCP connection per browser. Initialized lazily inside the lifespan.
_async_redis: Optional[redis_async.Redis] = None


@asynccontextmanager
async def lifespan(app_inst: FastAPI):
    """Construct shared async resources at startup and release them on shutdown."""
    global _async_redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _async_redis = redis_async.from_url(redis_url, decode_responses=True)
    try:
        yield
    finally:
        if _async_redis is not None:
            await _async_redis.aclose()
            _async_redis = None
        # Close the Postgres pool created lazily inside graph.py so SIGTERM
        # doesn't leak ~20 idle connections on every container restart.
        from graph import _pool as _graph_pool  # local import to avoid cycle
        if _graph_pool is not None:
            _graph_pool.close()


app = FastAPI(
    title="D&D Game Orchestrator (LangGraph)",
    description="Orchestrates game flow with LangGraph StateGraph, shared state, and Postgres-backed checkpointer persistence",
    version="3.1",
    lifespan=lifespan,
)

# Rate limiter: backed by Redis (see rate_limit.py). Default 100/min/IP +
# tighter per-route caps applied via @limiter.limit on hot handlers.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COMBAT_AGENT_URL = os.getenv("COMBAT_AGENT_URL", "http://localhost:9000")
ACTIVE_SESSIONS_KEY = "sessions:active"


def _register_session(session_id: str, metadata: dict) -> None:
    get_redis().hset(ACTIVE_SESSIONS_KEY, session_id, json.dumps(metadata))


def _unregister_session(session_id: str) -> bool:
    return bool(get_redis().hdel(ACTIVE_SESSIONS_KEY, session_id))


def _active_session_count() -> int:
    return int(get_redis().hlen(ACTIVE_SESSIONS_KEY))


# ========== Pydantic Models ==========
def _require_uuid(value: str) -> str:
    """Raise ValueError if `value` is not a UUID-shaped string."""
    try:
        UUID(value)
    except ValueError:
        raise ValueError("must be a valid UUID")
    return value


class PlayerInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    character_class: str = Field(min_length=1, max_length=50)
    character_name: str = Field(min_length=1, max_length=100)
    player_id: Optional[str] = None  # server-assigned if not provided

    @field_validator("player_id")
    @classmethod
    def _check_player_id(cls, v: Optional[str]) -> Optional[str]:
        return None if v is None else _require_uuid(v)


class UserInput(BaseModel):
    text: str
    session_id: Optional[str] = None
    player_id: Optional[str] = None  # required when party size > 1

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        if len(v) > 2000:
            raise ValueError("Action text must be 2000 characters or less")
        return v

    @field_validator("session_id", "player_id")
    @classmethod
    def _check_uuids(cls, v: Optional[str]) -> Optional[str]:
        return None if v is None else _require_uuid(v)


class CombatActionRequest(BaseModel):
    action: str


class GameStartRequest(BaseModel):
    campaign_id: Optional[str] = None
    # Solo path (legacy):
    character_class: Optional[str] = None
    character_name: Optional[str] = None
    # Multiplayer path:
    players: Optional[List[PlayerInput]] = None
    initial_prompt: Optional[str] = None
    max_combats: Optional[int] = 5
    combat_rounds: Optional[List[int]] = None

    @field_validator("players")
    @classmethod
    def validate_party_size(cls, v: Optional[List[PlayerInput]]) -> Optional[List[PlayerInput]]:
        if v is not None and len(v) > MAX_PARTY_SIZE:
            raise ValueError(f"Party size cannot exceed {MAX_PARTY_SIZE}")
        return v


def _resolve_party(request: GameStartRequest) -> list[PlayerCharacter]:
    """Build the party from either the multiplayer `players` list or the solo
    character_class/character_name pair. Always returns at least one PC."""
    if request.players:
        return [
            PlayerCharacter.from_class(
                character_class=p.character_class,
                character_name=p.character_name,
                name=p.name,
                player_id=p.player_id,
            )
            for p in request.players
        ]

    return [
        PlayerCharacter.from_class(
            character_class=request.character_class or "Fighter",
            character_name=request.character_name or "Adventurer",
            name=request.character_name or "Adventurer",
        )
    ]


def _build_initial_state(
    session_id: str,
    campaign_id: Optional[str],
    party: list[PlayerCharacter],
    max_combats: int,
    combat_rounds: Optional[List[int]],
    initial_prompt: str,
    campaign_metadata: dict,
) -> dict:
    """Build the initial DnDGameState for a new session."""
    return {
        "player_action": "",
        "pending_actions": {},
        "session_id": session_id,
        "state_type": "narration",
        "narration_round": 0,
        "combat_count": 0,
        "max_combats": max_combats,
        "combat_rounds": combat_rounds or [3, 5, 10, 15],
        "players": [pc.to_dict() for pc in party],
        "room_id": None,
        "acting_player_id": None,
        "initiative_order": [],
        "validation_result": {},
        "is_sabotage": False,
        "narrator_response": initial_prompt,
        "choices": [],
        "combat_response": {},
        "combat_session_id": None,
        "combat_trigger": "none",
        "campaign_id": campaign_id,
        "current_story_node_id": None,
        "campaign_metadata": campaign_metadata,
        "is_ending": False,
        "ending_type": None,
        "messages": [],
        "story_summary": "",
        "response": initial_prompt,
        "transition": None,
        "response_extras": {},
    }


def _format_response(session_id: str, state: dict) -> dict:
    extras = state.get("response_extras", {})

    response = {
        "session_id": session_id,
        "state_type": state.get("state_type", "narration"),
        "agent_used": _infer_agent(state),
        "response": state.get("response", ""),
        "validation": extras.get("validation"),
        "state_node": _build_state_node(state),
        "transition": state.get("transition"),
        "choices": state.get("choices", []),
        "narration_round": state.get("narration_round", 0),
        "combat_count": state.get("combat_count", 0),
        "max_combats": state.get("max_combats", 5),
        "players": state.get("players", []),
        "acting_player_id": state.get("acting_player_id"),
    }

    for key in ("combat_session_id", "combat_state", "combat_summary",
                "is_ending", "ending_type", "combat_available", "error"):
        if key in extras:
            response[key] = extras[key]

    if state.get("is_ending"):
        response["is_ending"] = True
        response["ending_type"] = state.get("ending_type")

    if state.get("combat_session_id"):
        response["combat_session_id"] = state["combat_session_id"]

    return response


def _infer_agent(state: dict) -> str:
    if state.get("is_sabotage"):
        return "orchestrator"
    if state.get("state_type") == "combat":
        return "combat"
    return "narrator"


def _build_state_node(state: dict) -> dict:
    return {
        "state_type": state.get("state_type", "narration"),
        "narrative_text": state.get("narrator_response", ""),
        "player_action": state.get("player_action", ""),
        "agent_response": state.get("response", ""),
        "metadata": state.get("campaign_metadata", {}),
        "narration_round": state.get("narration_round", 0),
        "combat_count": state.get("combat_count", 0),
    }


def _validate_uuid(value: str) -> None:
    try:
        UUID(value)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")


# ========== API Routes ==========
@app.get("/")
async def root():
    return {
        "service": "D&D Game Orchestrator (LangGraph)",
        "version": "3.1",
        "features": [
            "langgraph_state_machine",
            "postgres_checkpointer",
            "redis_session_index",
            "rule_validation",
            "multiplayer_state_shape",
        ],
    }


@app.get("/health")
def health_check():
    rule_agent_url = os.getenv("RULE_AGENT_URL", "http://localhost:9002")
    try:
        resp = requests.get(f"{rule_agent_url}/health", timeout=5)
        rule_agent_healthy = resp.status_code == 200
    except Exception:  # noqa: BLE001
        rule_agent_healthy = False

    try:
        redis_healthy = bool(get_redis().ping())
        active_sessions = _active_session_count()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Redis health check failed: {e}")
        redis_healthy = False
        active_sessions = 0

    return {
        "status": "healthy" if redis_healthy else "degraded",
        "services": {
            "rule_agent": rule_agent_healthy,
            "redis": redis_healthy,
            "active_sessions": active_sessions,
            "graph": "compiled",
        },
    }


@app.get("/campaigns")
def list_campaigns():
    return {"campaigns": CampaignLoader.list_campaigns()}


@app.get("/campaigns/{campaign_id}")
def get_campaign_details(campaign_id: str):
    campaign = CampaignLoader.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")
    return campaign.to_dict()


def _resolve_campaign(
    campaign_id: Optional[str],
    initial_prompt_override: Optional[str],
    lead: PlayerCharacter,
    party_size: int,
) -> tuple[str, dict, Optional[str]]:
    """Resolve a campaign id (or custom initial prompt) into the tuple
    (initial_prompt, campaign_metadata, current_story_node_id).

    Single source of truth shared by /game/start and /rooms/{id}/start. Raises
    HTTPException(400) when the campaign id is unknown.
    """
    initial_prompt = "Start a new D&D adventure in a fantasy tavern."
    campaign_metadata: dict = {}
    current_story_node_id: Optional[str] = None

    if campaign_id:
        try:
            campaign_data = CampaignLoader.initialize_campaign(
                campaign_id, lead.character_class, lead.character_name,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        initial_prompt = campaign_data["initial_prompt"]
        campaign_metadata = {
            "campaign_id": campaign_data["campaign_id"],
            "campaign_name": campaign_data["campaign_name"],
            "starting_location": campaign_data["starting_location"],
            "initial_quest": campaign_data["initial_quest"],
            **campaign_data["metadata"],
        }

        story_tree = StoryTreeLoader.load_story_tree(campaign_id)
        if story_tree:
            current_story_node_id = story_tree.root_node_id
            story_root = story_tree.get_root()
            if story_root:
                initial_prompt = story_root.narrative
                campaign_metadata["story_node_id"] = story_root.node_id
                campaign_metadata["story_choices"] = story_root.choices
                campaign_metadata["is_ending"] = story_root.is_ending
                campaign_metadata["combat_available"] = story_root.combat_available

        logger.info("Starting campaign: %s", campaign_metadata.get("campaign_name"))

    elif initial_prompt_override:
        initial_prompt = initial_prompt_override
        campaign_metadata = {"campaign_type": "custom", "party_size": party_size}
    else:
        campaign_metadata = {"campaign_type": "default", "party_size": party_size}

    return initial_prompt, campaign_metadata, current_story_node_id


@app.post("/game/start")
def start_game(request: GameStartRequest):
    """
    Initialize a new game session.

    Supports solo (`character_class`/`character_name`) or multiplayer (`players`).
    Session state is persisted via the Postgres checkpointer.
    """
    session_id = str(uuid4())
    party = _resolve_party(request)
    campaign_id = request.campaign_id
    initial_prompt, campaign_metadata, current_story_node_id = _resolve_campaign(
        campaign_id=campaign_id,
        initial_prompt_override=request.initial_prompt,
        lead=party[0],
        party_size=len(party),
    )

    initial_state = _build_initial_state(
        session_id=session_id,
        campaign_id=campaign_id,
        party=party,
        max_combats=request.max_combats or 5,
        combat_rounds=request.combat_rounds,
        initial_prompt=initial_prompt,
        campaign_metadata=campaign_metadata,
    )

    if current_story_node_id:
        initial_state["current_story_node_id"] = current_story_node_id

    initial_choices: list[str] = []
    try:
        initial_choices = generate_initial_choices(initial_prompt)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error generating initial choices: {e}")

    initial_state["choices"] = initial_choices

    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    graph.update_state(config, initial_state)

    _register_session(
        session_id,
        {"campaign_id": campaign_id, "party_size": len(party)},
    )

    logger.info(f"Started new game session: {session_id} (party of {len(party)})")

    combat_available = campaign_metadata.get("combat_available", False)

    return {
        "session_id": session_id,
        "state": _build_state_node(initial_state),
        "response": initial_prompt,
        "campaign_info": campaign_metadata,
        "choices": initial_choices,
        "is_ending": False,
        "combat_available": combat_available,
        "narration_round": 0,
        "combat_count": 0,
        "max_combats": request.max_combats or 5,
        "players": [pc.to_dict() for pc in party],
        "message": "Game started successfully!",
    }


@app.post("/game/action")
@limiter.limit(LIMIT_GAME_ACTION)
def game_action(request: Request, data: UserInput):
    """
    Handle a player action via the LangGraph state machine.

    For solo games, `player_id` is optional and defaults to the lone party
    member. For multiplayer use the room endpoints (Phase 3) which collect
    actions across the party before invoking the graph.
    """
    if not data.session_id:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new game first.")

    current_state = get_session_state(data.session_id)
    if not current_state:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new game first.")

    players = current_state.get("players", [])
    if not players:
        raise HTTPException(status_code=500, detail="Session has no party")

    if data.player_id is None:
        if len(players) > 1:
            raise HTTPException(status_code=400, detail="player_id is required for multiplayer sessions")
        player_id = players[0]["player_id"]
    else:
        player_id = data.player_id
        if not any(p["player_id"] == player_id for p in players):
            raise HTTPException(status_code=400, detail="player_id is not in this session's party")

    pending_actions = {player_id: data.text}
    combined = synthesize_party_action(players, pending_actions)

    logger.info(
        f"Session {data.session_id}: player {player_id} action in state "
        f"{current_state.get('state_type', 'unknown')}"
    )

    graph_input = _build_graph_input(
        session_id=data.session_id,
        current_state=current_state,
        combined_action=combined,
        pending_actions=pending_actions,
        players_dicts=players,
        room_id=current_state.get("room_id"),
    )

    try:
        result = invoke_game_action(data.session_id, combined, graph_input)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Graph invocation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing action. Please try again.")

    return _format_response(data.session_id, result)


@app.get("/game/state/{session_id}")
def get_game_state(session_id: str):
    state = get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = state.get("messages", [])
    story_parts = []
    for msg in messages[-10:]:
        content = msg.content if hasattr(msg, "content") else str(msg)
        if content:
            story_parts.append(content[:500])
    story_summary = "\n\n".join(story_parts)

    return {
        "session_id": session_id,
        "state_type": state.get("state_type", "narration"),
        "acting_player_id": state.get("acting_player_id"),
        "current_state": _build_state_node(state),
        "path": [],
        "story_summary": story_summary,
        "players": state.get("players", []),
        "full_tree": {
            "narration_round": state.get("narration_round", 0),
            "combat_count": state.get("combat_count", 0),
            "max_combats": state.get("max_combats", 5),
            "state_type": state.get("state_type", "narration"),
        },
    }


@app.get("/combat/state/{combat_session_id}")
def get_combat_state_proxy(combat_session_id: str):
    _validate_uuid(combat_session_id)
    try:
        response = requests.get(f"{COMBAT_AGENT_URL}/combat/state/{combat_session_id}", timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error getting combat state: {e}")
        raise HTTPException(status_code=500, detail="Failed to get combat state")


@app.post("/combat/action/{combat_session_id}")
def submit_combat_action(combat_session_id: str, action_data: CombatActionRequest):
    _validate_uuid(combat_session_id)
    try:
        response = requests.post(
            f"{COMBAT_AGENT_URL}/combat/action/{combat_session_id}",
            json={"action": action_data.action},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error submitting combat action: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit combat action")


@app.delete("/game/session/{session_id}")
def end_game_session(session_id: str):
    _validate_uuid(session_id)
    _unregister_session(session_id)

    logger.info(f"Ended game session: {session_id}")
    return {"message": "Game session ended", "session_id": session_id}


# ========================================================================
# Multiplayer rooms
# ========================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ROUND_TIMEOUT_SECONDS = int(os.getenv("ROOM_ROUND_TIMEOUT", "30"))
SSE_HEARTBEAT_SECONDS = 15

# Per-process timer registry. Single-replica safe; if you scale the
# orchestrator horizontally, move this to Redis (a sorted-set scheduler).
_round_timers: Dict[tuple[str, int], asyncio.Task] = {}


class CreateRoomRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    host: PlayerInput
    max_players: Optional[int] = ROOM_MAX_PARTY_SIZE


class JoinRoomRequest(BaseModel):
    player: PlayerInput


class LeaveRoomRequest(BaseModel):
    player_id: str

    @field_validator("player_id")
    @classmethod
    def _check_player_id(cls, v: str) -> str:
        return _require_uuid(v)


class StartRoomRequest(BaseModel):
    player_id: str  # must be the host
    campaign_id: Optional[str] = None
    initial_prompt: Optional[str] = None
    max_combats: Optional[int] = 5
    combat_rounds: Optional[List[int]] = None

    @field_validator("player_id")
    @classmethod
    def _check_player_id(cls, v: str) -> str:
        return _require_uuid(v)


class RoomActionRequest(BaseModel):
    player_id: str
    text: str = Field(max_length=2000)

    @field_validator("player_id")
    @classmethod
    def _check_player_id(cls, v: str) -> str:
        return _require_uuid(v)


def _player_input_to_pc(p: PlayerInput) -> PlayerCharacter:
    return PlayerCharacter.from_class(
        character_class=p.character_class,
        character_name=p.character_name,
        name=p.name,
        player_id=p.player_id,
    )


def _room_response(room: Room) -> dict:
    players = RoomManager.list_players(room.room_id)
    return {
        **room.to_dict(),
        "players": [pc.to_dict() for pc in players],
    }


def _validate_room_id(room_id: str) -> None:
    try:
        UUID(room_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid room ID format")


def _require_player_token(room_id: str, player_id: str, token: Optional[str]) -> None:
    """Reject the request if the X-Player-Token doesn't match the stored token.

    Raises HTTP 401 (no token) or 403 (wrong token / unknown player). The
    distinction is intentional: 401 prompts the client to obtain a token,
    403 means we have one and it's wrong.
    """
    if not token:
        raise HTTPException(status_code=401, detail="X-Player-Token header required")
    if not RoomManager.verify_token(room_id, player_id, token):
        raise HTTPException(status_code=403, detail="invalid player token")


@app.post("/rooms")
@limiter.limit(LIMIT_CREATE_ROOM)
def create_room(request: Request, req: CreateRoomRequest):
    if req.max_players and req.max_players > ROOM_MAX_PARTY_SIZE:
        raise HTTPException(status_code=400, detail=f"max_players cannot exceed {ROOM_MAX_PARTY_SIZE}")

    host_pc = _player_input_to_pc(req.host)
    try:
        room = RoomManager.create_room(req.name, host_pc.player_id, req.max_players or ROOM_MAX_PARTY_SIZE)
        RoomManager.add_player(room.room_id, host_pc)
    except RoomError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Issue the host's token. Returned ONCE here — the client must store it
    # (sessionStorage) and present it via X-Player-Token on every subsequent
    # mutating request. Tokens are never echoed in SSE broadcasts.
    token = RoomManager.issue_token(room.room_id, host_pc.player_id)

    return {
        **_room_response(RoomManager.get_room(room.room_id)),
        "player_id": host_pc.player_id,
        "player_token": token,
    }


@app.get("/rooms")
def list_rooms():
    return {"rooms": [_room_response(r) for r in RoomManager.list_rooms()]}


@app.get("/rooms/{room_id}")
def get_room(room_id: str):
    _validate_room_id(room_id)
    room = RoomManager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return _room_response(room)


@app.post("/rooms/{room_id}/join")
def join_room(room_id: str, req: JoinRoomRequest):
    _validate_room_id(room_id)
    pc = _player_input_to_pc(req.player)
    try:
        RoomManager.add_player(room_id, pc)
    except RoomError as e:
        raise HTTPException(status_code=400, detail=str(e))
    room = RoomManager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    token = RoomManager.issue_token(room_id, pc.player_id)
    return {
        "player_id": pc.player_id,
        "player_token": token,
        "room": _room_response(room),
    }


@app.post("/rooms/{room_id}/leave")
def leave_room(
    room_id: str,
    req: LeaveRoomRequest,
    x_player_token: Optional[str] = Header(default=None, alias="X-Player-Token"),
):
    _validate_room_id(room_id)
    _require_player_token(room_id, req.player_id, x_player_token)
    room = RoomManager.remove_player(room_id, req.player_id)
    if room is None:
        return {"message": "left", "room": None}
    return {"message": "left", "room": _room_response(room)}


@app.post("/rooms/{room_id}/start")
def start_room(
    room_id: str,
    req: StartRoomRequest,
    x_player_token: Optional[str] = Header(default=None, alias="X-Player-Token"),
):
    """Host transitions the room from lobby to active and creates the LangGraph thread.
    The room_id IS the LangGraph thread_id from this point on.
    """
    _validate_room_id(room_id)
    _require_player_token(room_id, req.player_id, x_player_token)
    room = RoomManager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    if room.state != ROOM_STATE_LOBBY:
        raise HTTPException(status_code=400, detail="Room already started")
    if room.host_player_id != req.player_id:
        raise HTTPException(status_code=403, detail="Only the host can start the room")

    party = RoomManager.list_players(room_id)
    if not party:
        raise HTTPException(status_code=400, detail="Room has no players")

    initial_prompt, campaign_metadata, current_story_node_id = _resolve_campaign(
        campaign_id=req.campaign_id,
        initial_prompt_override=req.initial_prompt,
        lead=party[0],
        party_size=len(party),
    )

    # Seed the graph thread (room_id == thread_id)
    initial_state = _build_initial_state(
        session_id=room_id,
        campaign_id=req.campaign_id,
        party=party,
        max_combats=req.max_combats or 5,
        combat_rounds=req.combat_rounds,
        initial_prompt=initial_prompt,
        campaign_metadata=campaign_metadata,
    )
    initial_state["room_id"] = room_id
    if current_story_node_id:
        initial_state["current_story_node_id"] = current_story_node_id

    initial_choices: list[str] = []
    try:
        initial_choices = generate_initial_choices(initial_prompt)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error generating initial choices: {e}")
    initial_state["choices"] = initial_choices

    graph = get_graph()
    config = {"configurable": {"thread_id": room_id}}
    graph.update_state(config, initial_state)

    RoomManager.mark_active(room_id, req.campaign_id)
    _register_session(room_id, {"campaign_id": req.campaign_id, "party_size": len(party), "room_id": room_id})

    return {
        "room_id": room_id,
        "session_id": room_id,
        "response": initial_prompt,
        "campaign_info": campaign_metadata,
        "choices": initial_choices,
        "players": [pc.to_dict() for pc in party],
        "max_combats": req.max_combats or 5,
        "current_round": 0,
    }


@app.post("/rooms/{room_id}/action")
@limiter.limit(LIMIT_ROOM_ACTION)
async def submit_room_action(
    request: Request,
    room_id: str,
    req: RoomActionRequest,
    x_player_token: Optional[str] = Header(default=None, alias="X-Player-Token"),
):
    """Collect a player's action for the current round.

    Narration: wait until every player has submitted (or ROUND_TIMEOUT_SECONDS
    elapses), then resolve. Combat: only `acting_player_id` may submit, and
    the round resolves immediately on that single submission so the next
    PC's turn can begin.
    """
    _validate_room_id(room_id)
    _require_player_token(room_id, req.player_id, x_player_token)
    room = RoomManager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    if room.state != ROOM_STATE_ACTIVE:
        raise HTTPException(status_code=400, detail="Room is not active")

    party = RoomManager.list_players(room_id)
    if not any(pc.player_id == req.player_id for pc in party):
        raise HTTPException(status_code=400, detail="player is not in this room")

    state = get_session_state(room_id)
    if not state:
        raise HTTPException(status_code=500, detail="Room session state missing")
    state_type = state.get("state_type", "narration")
    round_no = room.current_round

    if state_type == "combat":
        acting_id = state.get("acting_player_id")
        if acting_id and req.player_id != acting_id:
            raise HTTPException(status_code=403, detail="not your turn")
        actions = {req.player_id: req.text}
        result = await _resolve_round(room_id, round_no, party, actions)
        return {
            "resolved": True,
            "submitted": [req.player_id],
            "result": result,
        }

    # Narration: collect and wait for full party (or timeout)
    try:
        actions = RoomManager.submit_action(room_id, round_no, req.player_id, req.text)
    except RoomError as e:
        raise HTTPException(status_code=400, detail=str(e))

    submitted = set(actions.keys())
    party_ids = {pc.player_id for pc in party}

    if submitted >= party_ids:
        result = await _resolve_round(room_id, round_no, party, actions)
        return {"resolved": True, "submitted": list(submitted), "result": result}

    _schedule_round_timeout(room_id, round_no)
    return {"resolved": False, "submitted": list(submitted), "expecting": len(party_ids)}


def _schedule_round_timeout(room_id: str, round_no: int) -> None:
    """Caller is an async handler, so the running loop is already current.
    Use create_task directly — get_event_loop is deprecated in 3.10+."""
    key = (room_id, round_no)
    if key in _round_timers and not _round_timers[key].done():
        return
    _round_timers[key] = asyncio.create_task(_round_timeout_task(room_id, round_no))


def _cancel_round_timeout(room_id: str, round_no: int) -> None:
    key = (room_id, round_no)
    task = _round_timers.pop(key, None)
    if task and not task.done():
        task.cancel()


async def _round_timeout_task(room_id: str, round_no: int) -> None:
    try:
        await asyncio.sleep(ROUND_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        return

    # Sync Redis/RoomManager work happens off the event loop.
    party = await asyncio.to_thread(RoomManager.list_players, room_id)
    actions = await asyncio.to_thread(
        lambda: get_redis().hgetall(f"room:{room_id}:actions:{round_no}") or {}
    )
    if len(actions) == len(party):
        return  # already resolved by another path

    logger.info("Round %d in room %s timed out; auto-resolving", round_no, room_id)
    for pc in party:
        actions.setdefault(pc.player_id, "holds action and observes the situation")
    await _resolve_round(room_id, round_no, party, actions)


def _build_graph_input(
    session_id: str,
    current_state: dict,
    *,
    combined_action: str,
    pending_actions: dict[str, str],
    players_dicts: list[dict],
    room_id: Optional[str],
) -> dict:
    """Single source of truth for the graph_input dict shape.

    Used by both single-player /game/action and multiplayer round resolution.
    Centralizing this prevents silent state drift if a key is added in one
    call site but missed in the other (LangGraph would silently fall back to
    the prior checkpoint value).
    """
    return {
        "player_action": combined_action,
        "pending_actions": pending_actions,
        "session_id": session_id,
        "state_type": current_state.get("state_type", "narration"),
        "narration_round": current_state.get("narration_round", 0),
        "combat_count": current_state.get("combat_count", 0),
        "max_combats": current_state.get("max_combats", 5),
        "combat_rounds": current_state.get("combat_rounds", [3, 5, 10, 15]),
        "players": players_dicts,
        "room_id": room_id,
        "acting_player_id": current_state.get("acting_player_id"),
        "initiative_order": current_state.get("initiative_order", []),
        "combat_session_id": current_state.get("combat_session_id"),
        "campaign_id": current_state.get("campaign_id"),
        "current_story_node_id": current_state.get("current_story_node_id"),
        "story_summary": current_state.get("story_summary", ""),
        "campaign_metadata": current_state.get("campaign_metadata", {}),
        "is_ending": False,
        "ending_type": None,
        "combat_trigger": "none",
        "transition": None,
        "response_extras": {},
        "messages": current_state.get("messages", []),
        "validation_result": {},
        "is_sabotage": False,
        "narrator_response": "",
        "choices": [],
        "combat_response": {},
        "response": "",
    }


async def _resolve_round(
    room_id: str,
    round_no: int,
    party: list[PlayerCharacter],
    actions: dict[str, str],
) -> Optional[dict]:
    """Acquire the per-room lock, invoke the graph with collected actions,
    advance the round counter, and broadcast the result to subscribers.

    Background-task safe: never raises HTTPException (runs from a timer
    coroutine outside any HTTP request context).
    """
    if not await asyncio.to_thread(RoomManager.acquire_invoke_lock, room_id):
        logger.info("Another worker is resolving round %d in room %s", round_no, room_id)
        return None
    try:
        current_state = await asyncio.to_thread(get_session_state, room_id)
        if not current_state:
            logger.error("Room session state missing for %s", room_id)
            return None

        players_dicts = [pc.to_dict() for pc in party]
        combined = synthesize_party_action(players_dicts, actions)

        graph_input = _build_graph_input(
            session_id=room_id,
            current_state=current_state,
            combined_action=combined,
            pending_actions=dict(actions),
            players_dicts=players_dicts,
            room_id=room_id,
        )

        result = await asyncio.to_thread(invoke_game_action, room_id, combined, graph_input)

        new_round = await asyncio.to_thread(RoomManager.increment_round, room_id)
        await asyncio.to_thread(RoomManager.clear_round_actions, room_id, round_no)
        _cancel_round_timeout(room_id, round_no)

        formatted = _format_response(room_id, result)
        formatted["current_round"] = new_round

        # Async publish so we don't block the event loop on fanout.
        if _async_redis is not None:
            try:
                await _async_redis.publish(
                    events_channel(room_id),
                    json.dumps({
                        "type": "round-resolved",
                        "round": round_no,
                        "result": formatted,
                    }),
                )
            except Exception as e:  # noqa: BLE001 - pub/sub is best-effort
                logger.warning("Failed to publish round-resolved for %s: %s", room_id, e)

        if result.get("state_type") == "game_over" or result.get("is_ending"):
            await asyncio.to_thread(RoomManager.mark_ended, room_id)

        return formatted
    finally:
        await asyncio.to_thread(RoomManager.release_invoke_lock, room_id)


@app.get("/rooms/{room_id}/events")
@limiter.limit(LIMIT_SSE_CONNECT)
async def stream_room_events(request: Request, room_id: str):
    """SSE stream of room events. Client uses EventSource."""
    _validate_room_id(room_id)
    if not RoomManager.get_room(room_id):
        raise HTTPException(status_code=404, detail="Room not found")
    if _async_redis is None:
        raise HTTPException(status_code=503, detail="Event stream not yet initialized")

    async def event_stream() -> AsyncIterator[str]:
        # Reuse the shared async client; only the pubsub object is per-connection.
        pubsub = _async_redis.pubsub()
        await pubsub.subscribe(events_channel(room_id))

        try:
            # Initial snapshot so the client doesn't have to poll
            room = RoomManager.get_room(room_id)
            if room:
                yield f"data: {json.dumps({'type': 'snapshot', 'room': _room_response(room)})}\n\n"

            while True:
                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                        timeout=SSE_HEARTBEAT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue

                if msg is None:
                    yield ": heartbeat\n\n"
                    continue
                if msg.get("type") != "message":
                    continue
                yield f"data: {msg['data']}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await pubsub.unsubscribe()
                await pubsub.aclose()
                # Don't close the shared _async_redis here — it's used by other
                # connections.
            except Exception as e:  # noqa: BLE001 - cleanup is best-effort
                logger.warning("SSE pubsub cleanup error for %s: %s", room_id, e)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
