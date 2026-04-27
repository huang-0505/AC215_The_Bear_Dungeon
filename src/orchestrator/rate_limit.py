"""
rate_limit.py

slowapi Limiter configured with Redis storage so multiple orchestrator
replicas (when we get there) share a single rate-limit window.

Today's limits are per-IP since there's no auth identity to key on outside
the multiplayer rooms. Once player tokens are widespread the limits should
be tightened to per-player.
"""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address


# Use the same Redis we run for rooms / sessions.
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Default cap covers any unmarked endpoint. Stricter per-route limits below.
DEFAULT_LIMITS = ["100/minute"]

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=DEFAULT_LIMITS,
    storage_uri=_REDIS_URL,
    headers_enabled=True,  # add X-RateLimit-* response headers
)


# Per-route caps reused across handler decorators.
LIMIT_CREATE_ROOM = "10/minute"
LIMIT_ROOM_ACTION = "60/minute"
LIMIT_GAME_ACTION = "60/minute"
LIMIT_SSE_CONNECT = "30/minute"
