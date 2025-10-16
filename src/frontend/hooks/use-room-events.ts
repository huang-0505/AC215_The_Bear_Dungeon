"use client"

import { useEffect, useState, useRef } from "react"
import type { GameRoom } from "@/lib/multiplayer/types"

export function useRoomEvents(roomId: string | null) {
  const [room, setRoom] = useState<GameRoom | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!roomId) {
      setRoom(null)
      setIsConnected(false)
      return
    }

    console.log(`[v0] Connecting to room events: ${roomId}`)

    const eventSource = new EventSource(`/api/rooms/${roomId}/events`)
    eventSourceRef.current = eventSource

    eventSource.onopen = () => {
      console.log(`[v0] SSE connected to room ${roomId}`)
      setIsConnected(true)
    }

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log(`[v0] SSE message:`, data.type)

        if (data.type === "room-update" && data.room) {
          setRoom(data.room)
        }
      } catch (error) {
        console.error("[v0] Error parsing SSE message:", error)
      }
    }

    eventSource.onerror = (error) => {
      console.error(`[v0] SSE error for room ${roomId}:`, error)
      setIsConnected(false)
    }

    return () => {
      console.log(`[v0] Disconnecting from room events: ${roomId}`)
      eventSource.close()
      eventSourceRef.current = null
      setIsConnected(false)
    }
  }, [roomId])

  return { room, isConnected }
}
