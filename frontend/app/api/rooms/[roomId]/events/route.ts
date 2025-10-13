import type { NextRequest } from "next/server"
import { roomManager } from "@/lib/multiplayer/room-manager"
import { roomEventEmitter } from "@/lib/multiplayer/event-emitter"

// GET /api/rooms/[roomId]/events - Server-Sent Events for real-time updates
export async function GET(request: NextRequest, { params }: { params: { roomId: string } }) {
  const { roomId } = params

  // Verify room exists
  const room = roomManager.getRoom(roomId)
  if (!room) {
    return new Response("Room not found", { status: 404 })
  }

  // Create SSE stream
  const stream = new ReadableStream({
    start(controller) {
      console.log(`[v0] SSE connection opened for room ${roomId}`)

      // Send initial room state
      const encoder = new TextEncoder()
      const sendEvent = (data: any) => {
        const message = `data: ${JSON.stringify(data)}\n\n`
        controller.enqueue(encoder.encode(message))
      }

      // Send current room state immediately
      sendEvent({
        type: "room-update",
        room: roomManager.getRoom(roomId),
      })

      // Subscribe to room updates
      const unsubscribe = roomEventEmitter.subscribe(roomId, (data) => {
        sendEvent(data)
      })

      // Send heartbeat every 30 seconds
      const heartbeat = setInterval(() => {
        sendEvent({ type: "heartbeat", timestamp: Date.now() })
      }, 30000)

      // Cleanup on close
      request.signal.addEventListener("abort", () => {
        console.log(`[v0] SSE connection closed for room ${roomId}`)
        clearInterval(heartbeat)
        unsubscribe()
      })
    },
  })

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Cache-Control",
    },
  })
}
