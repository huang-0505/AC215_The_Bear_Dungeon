"""
graph.py

LangGraph StateGraph definition for the D&D game orchestrator.
Defines the graph structure, compiles it with a Postgres-backed checkpointer,
and provides the public interface for invoking the graph.
"""

import os
import logging
import threading

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from graph_state import DnDGameState
from nodes.rule_validator_node import rule_validator_node
from nodes.router_node import router_node
from nodes.narrator_node import narrator_node
from nodes.combat_node import combat_node
from nodes.game_over_node import game_over_node
from nodes.response_builder_node import response_builder_node
from nodes.reject_node import reject_node
from nodes.summarizer_node import summarizer_node

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is required. Set it in .env or your container env. "
        "Example: postgresql://dnd:<password>@postgres:5432/dnd"
    )


def _build_graph() -> StateGraph:
    """
    Build the D&D game orchestrator graph.

    Graph structure:
        START -> rule_validator -> (conditional: router_node)
            -> "reject"   -> reject_node -> END
            -> "game_over" -> game_over_node -> response_builder -> END
            -> "narrator"  -> narrator_node -> post_narrator_router
                -> "combat" (ai_detected) -> combat_node -> response_builder -> summarizer -> END
                -> "continue" -> response_builder -> summarizer -> END
            -> "combat"   -> combat_node -> response_builder -> summarizer -> END
    """
    builder = StateGraph(DnDGameState)

    # --- Register nodes ---
    builder.add_node("rule_validator", rule_validator_node)
    builder.add_node("narrator", narrator_node)
    builder.add_node("combat", combat_node)
    builder.add_node("game_over", game_over_node)
    builder.add_node("response_builder", response_builder_node)
    builder.add_node("reject", reject_node)
    builder.add_node("summarizer", summarizer_node)

    # --- Entry point ---
    builder.set_entry_point("rule_validator")

    # --- Conditional edge: rule_validator -> router ---
    builder.add_conditional_edges(
        "rule_validator",
        router_node,
        {
            "reject": "reject",
            "game_over": "game_over",
            "narrator": "narrator",
            "combat": "combat",
        },
    )

    # --- Post-narrator routing: check for AI-detected combat trigger ---
    def post_narrator_router(state: DnDGameState) -> str:
        """After narration, check if AI detected combat or if game ended."""
        if state.get("state_type") == "game_over":
            return "game_over"
        if state.get("combat_trigger") == "ai_detected":
            return "combat"
        return "continue"

    builder.add_conditional_edges(
        "narrator",
        post_narrator_router,
        {
            "game_over": "game_over",
            "combat": "combat",
            "continue": "response_builder",
        },
    )

    # --- Simple edges ---
    builder.add_edge("combat", "response_builder")
    builder.add_edge("game_over", "response_builder")
    builder.add_edge("reject", "response_builder")
    builder.add_edge("response_builder", "summarizer")
    builder.add_edge("summarizer", END)

    return builder


# --- Module-level graph singleton ---
_graph_builder = _build_graph()
_pool: ConnectionPool | None = None
_checkpointer: PostgresSaver | None = None
_compiled_graph = None
_graph_lock = threading.Lock()


def get_graph():
    """Get the compiled graph with Postgres checkpointer (thread-safe lazy init)."""
    global _pool, _checkpointer, _compiled_graph

    if _compiled_graph is not None:
        return _compiled_graph

    with _graph_lock:
        if _compiled_graph is not None:
            return _compiled_graph

        # Sizing rationale: each LangGraph turn issues ~5 reads + ~3 writes
        # against the checkpointer. At 50 concurrent rooms with one in-flight
        # turn each, peak demand is ~50 connections; max_size=50 with
        # min_size=5 keeps a warm baseline. `timeout` makes saturated pools
        # fail loudly instead of hanging requests.
        _pool = ConnectionPool(
            conninfo=DATABASE_URL,
            min_size=5,
            max_size=50,
            timeout=30,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        _checkpointer = PostgresSaver(_pool)
        _checkpointer.setup()  # creates checkpoint tables on first run; idempotent

        _compiled_graph = _graph_builder.compile(checkpointer=_checkpointer)
        logger.info("LangGraph compiled with Postgres checkpointer")

    return _compiled_graph


def invoke_game_action(session_id: str, player_action: str, current_state: dict) -> dict:
    """
    Invoke the game graph for a player action.

    Args:
        session_id: The game session ID (used as thread_id for checkpointing)
        player_action: The player's action text
        current_state: Current game state values to merge into the graph input

    Returns:
        The final graph state after processing
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    input_state = {
        "player_action": player_action,
        "session_id": session_id,
        **current_state,
    }

    return graph.invoke(input_state, config)


def get_session_state(session_id: str) -> dict | None:
    """
    Retrieve the latest checkpointed state for a session.

    Args:
        session_id: The game session ID

    Returns:
        The state dict, or None if session not found
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            return snapshot.values
    except Exception as e:  # noqa: BLE001 - upstream callers expect None on miss
        logger.error(f"Error getting session state: {e}")

    return None
