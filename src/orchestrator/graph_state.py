"""
graph_state.py

Shared state definition for the LangGraph-based D&D game orchestrator.
All graph nodes read from and write to this TypedDict.

State is multiplayer-shaped: `players` is the party (1..N entries),
`pending_actions` collects each player's submitted action for the current
round before the graph is invoked, and `acting_player_id` identifies whose
turn it is during initiative-ordered combat.

For backward-compat with single-player request flow, `player_action` holds
the synthesized "what the party is doing" string the rule validator and
narrator consume.
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class DnDGameState(TypedDict):
    """Shared state passed between all LangGraph nodes."""

    # --- Player input ---
    player_action: str                      # combined party action text (synthesized)
    pending_actions: dict[str, str]         # player_id -> raw action text for this round
    session_id: str                         # LangGraph thread_id; equals room_id when in a room

    # --- Game tracking ---
    state_type: str                         # "narration" | "combat" | "game_over"
    narration_round: int
    combat_count: int
    max_combats: int
    combat_rounds: list[int]

    # --- Party / multiplayer ---
    players: list[dict]                     # PlayerCharacter dicts; len in [1, 4]
    room_id: Optional[str]                  # set when this session is bound to a multiplayer room

    # --- Combat turn ordering ---
    acting_player_id: Optional[str]         # whose turn during combat
    initiative_order: list[str]             # ordered list of player_id + enemy_id

    # --- Rule validation output ---
    validation_result: dict
    is_sabotage: bool

    # --- Narrator output ---
    narrator_response: str
    choices: list[str]

    # --- Combat state ---
    combat_response: dict
    combat_session_id: Optional[str]
    combat_trigger: str                     # "none" | "round_based" | "player_requested" | "ai_detected"

    # --- Story tree ---
    campaign_id: Optional[str]
    current_story_node_id: Optional[str]

    # --- Game metadata ---
    campaign_metadata: dict
    is_ending: bool
    ending_type: Optional[str]

    # --- Conversation history (append-only via add_messages reducer) ---
    messages: Annotated[list, add_messages]

    # --- Story summary for narrator context ---
    story_summary: str

    # --- Final response to send to frontend ---
    response: str
    transition: Optional[str]
    response_extras: dict
