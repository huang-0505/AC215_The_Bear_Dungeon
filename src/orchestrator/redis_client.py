"""
redis_client.py

Singleton Redis client used across the orchestrator for ephemeral state:
multiplayer rooms, presence, action collection, pub/sub fanout, and the
lightweight active-session index.

Durable game state lives in Postgres via the LangGraph checkpointer (graph.py).
"""

import os
import threading
from typing import Optional

import redis

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_client: Optional[redis.Redis] = None
_lock = threading.Lock()


def get_redis() -> redis.Redis:
    """Return the shared Redis client (thread-safe lazy init)."""
    global _client

    if _client is not None:
        return _client

    with _lock:
        if _client is not None:
            return _client
        _client = redis.Redis.from_url(
            _REDIS_URL,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            health_check_interval=30,
        )

    return _client


def reset_for_tests() -> None:
    """Drop the cached client. Tests use this to swap in a fakeredis instance."""
    global _client
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:  # noqa: BLE001 - shutdown best-effort
                pass
        _client = None
