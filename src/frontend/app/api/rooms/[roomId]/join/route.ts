import { type NextRequest, NextResponse } from "next/server"
import { roomManager } from "@/lib/multiplayer/room-manager"

// POST /api/rooms/[roomId]/join - Join a room
export async function POST(request: NextRequest, { params }: { params: { roomId: string } }) {
  try {
    const { playerId, playerName, characterClass } = await request.json()

    if (!playerId || !playerName || !characterClass) {
      return NextResponse.json({ error: "Player ID, name, and character class are required" }, { status: 400 })
    }

    const room = roomManager.joinRoom(params.roomId, {
      id: playerId,
      name: playerName,
      characterClass,
      isReady: false,
    })

    if (!room) {
      return NextResponse.json({ error: "Room not found" }, { status: 404 })
    }

    return NextResponse.json({ room })
  } catch (error) {
    console.error("[v0] Error joining room:", error)
    const message = error instanceof Error ? error.message : "Failed to join room"
    return NextResponse.json({ error: message }, { status: 400 })
  }
}
