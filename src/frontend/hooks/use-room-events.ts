"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import type {
  GameStateSnapshot,
  Room,
  RoomEvent,
  RoundResult,
} from "@/lib/multiplayer/types"

interface RoomEventsState {
  room: Room | null
  isConnected: boolean
  submittedPlayerIds: string[]
  lastResult: RoundResult | null
  isEnded: boolean
  // Derived from the latest round-resolved event; null while in narration or
  // before any round has resolved. Use it to gate the action input on the
  // acting player's UI in combat.
  actingPlayerId: string | null
  stateType: "narration" | "combat" | "game_over" | null
}

const initialState: RoomEventsState = {
  room: null,
  isConnected: false,
  submittedPlayerIds: [],
  lastResult: null,
  isEnded: false,
  actingPlayerId: null,
  stateType: null,
}

/**
 * Subscribe to a room's SSE event stream and maintain a derived snapshot
 * of room state. Initial snapshot arrives as the first event from the
 * server; subsequent events mutate the snapshot in place.
 */
export function useRoomEvents(roomId: string | null) {
  const [state, setState] = useState<RoomEventsState>(initialState)
  const eventSourceRef = useRef<EventSource | null>(null)
  // Guard so we only seed game state once per room session — without this,
  // every snapshot event (re-broadcast on join/leave/start) would refire
  // the fetch and race with itself.
  const seededRef = useRef(false)

  const refreshFromServer = useCallback(async () => {
    if (!roomId) return
    try {
      const response = await fetch(`/api/rooms/${roomId}`)
      if (!response.ok) return
      const room = (await response.json()) as Room
      setState((prev) => ({ ...prev, room }))
    } catch (error) {
      // Best-effort refresh; logging only avoids leaking PII into UI.
      // eslint-disable-next-line no-console
      console.warn("Room refresh failed", error)
    }
  }, [roomId])

  const seedGameState = useCallback(async () => {
    if (!roomId || seededRef.current) return
    seededRef.current = true
    try {
      const response = await fetch(`/api/game/state/${roomId}`)
      if (!response.ok) {
        // The host may not have started yet — clear the guard so we retry
        // when room-started fires.
        seededRef.current = false
        return
      }
      const snapshot = (await response.json()) as GameStateSnapshot
      setState((prev) => ({
        ...prev,
        actingPlayerId: snapshot.acting_player_id,
        stateType: snapshot.state_type,
      }))
    } catch {
      seededRef.current = false
    }
  }, [roomId])

  useEffect(() => {
    if (!roomId) {
      setState(initialState)
      return
    }

    const source = new EventSource(`/api/rooms/${roomId}/events`)
    eventSourceRef.current = source

    source.onopen = () => {
      setState((prev) => ({ ...prev, isConnected: true }))
    }

    source.onerror = () => {
      setState((prev) => ({ ...prev, isConnected: false }))
    }

    source.onmessage = (event) => {
      let parsed: RoomEvent
      try {
        parsed = JSON.parse(event.data) as RoomEvent
      } catch {
        return
      }

      setState((prev) => applyEvent(prev, parsed))

      // For player join/leave we don't get a fresh full snapshot — fetch one.
      if (
        parsed.type === "player-joined" ||
        parsed.type === "player-left" ||
        parsed.type === "room-started"
      ) {
        void refreshFromServer()
      }

      // After the host starts the room, the LangGraph thread now exists —
      // fetch the seed game state so we know whose turn it is.
      if (parsed.type === "snapshot" && parsed.room.state === "active") {
        void seedGameState()
      }
      if (parsed.type === "room-started") {
        void seedGameState()
      }
    }

    return () => {
      source.close()
      eventSourceRef.current = null
      seededRef.current = false
      setState(initialState)
    }
  }, [roomId, refreshFromServer, seedGameState])

  return state
}

function applyEvent(prev: RoomEventsState, event: RoomEvent): RoomEventsState {
  switch (event.type) {
    case "snapshot":
      return { ...prev, room: event.room, isConnected: true }
    case "action-submitted":
      return { ...prev, submittedPlayerIds: event.submitted }
    case "round-resolved":
      return {
        ...prev,
        lastResult: event.result,
        submittedPlayerIds: [],
        actingPlayerId: event.result.acting_player_id,
        stateType: event.result.state_type,
        room: prev.room
          ? {
              ...prev.room,
              current_round: event.result.current_round,
              players: event.result.players ?? prev.room.players,
            }
          : prev.room,
        isEnded: Boolean(event.result.is_ending),
      }
    case "room-ended":
    case "room-deleted":
      return { ...prev, isEnded: true }
    case "room-started":
      return prev
    case "player-joined":
    case "player-left":
      return prev
    default:
      return prev
  }
}
