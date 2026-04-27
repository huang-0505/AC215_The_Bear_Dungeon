// Shapes mirror the orchestrator's room and player payloads (see
// src/orchestrator/rooms.py and src/orchestrator/models/player.py).

export type RoomState = "lobby" | "active" | "ended"

export interface PlayerCharacter {
  player_id: string
  name: string
  character_class: string
  character_name: string
  hp: number
  max_hp: number
  ac: number
  attack_bonus: number
  damage: number
  attributes: Record<string, number>
  is_alive: boolean
}

export interface Room {
  room_id: string
  name: string
  state: RoomState
  host_player_id: string | null
  campaign_id: string | null
  max_players: number
  created_at: number
  current_round: number
  players: PlayerCharacter[]
}

// Events the orchestrator publishes over Redis pub/sub and forwards via SSE.
export type RoomEvent =
  | { type: "snapshot"; room: Room }
  | { type: "player-joined"; player_id: string }
  | { type: "player-left"; player_id: string }
  | { type: "room-started"; campaign_id: string | null }
  | { type: "room-ended" }
  | { type: "room-deleted" }
  | {
      type: "action-submitted"
      player_id: string
      round: number
      submitted: string[]
    }
  | { type: "round-resolved"; round: number; result: RoundResult }

export interface RoundResult {
  session_id: string
  state_type: "narration" | "combat" | "game_over"
  agent_used: string
  response: string
  choices: string[]
  current_round: number
  narration_round: number
  combat_count: number
  max_combats: number
  players: PlayerCharacter[]
  acting_player_id: string | null
  is_ending?: boolean
  ending_type?: string
  combat_session_id?: string
  combat_state?: unknown
  combat_summary?: unknown
}

// Snapshot returned by GET /api/game/state/{session_id} — used by the game
// page on mount to know whose turn it is before any round resolves.
export interface GameStateSnapshot {
  session_id: string
  state_type: "narration" | "combat" | "game_over"
  acting_player_id: string | null
  current_state: {
    state_type: "narration" | "combat" | "game_over"
    narrative_text: string
    player_action: string
    agent_response: string
    metadata: Record<string, unknown>
    narration_round: number
    combat_count: number
  }
  story_summary: string
  players: PlayerCharacter[]
  full_tree: {
    narration_round: number
    combat_count: number
    max_combats: number
    state_type: "narration" | "combat" | "game_over"
  }
}

export interface PlayerInput {
  name: string
  character_class: string
  character_name: string
  player_id?: string
}
