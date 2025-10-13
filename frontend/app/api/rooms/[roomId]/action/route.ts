import { type NextRequest, NextResponse } from "next/server"
import { roomManager } from "@/lib/multiplayer/room-manager"

// POST /api/rooms/[roomId]/action - Submit player action
export async function POST(request: NextRequest, { params }: { params: { roomId: string } }) {
  try {
    const { playerId, action } = await request.json()

    if (!playerId || !action) {
      return NextResponse.json({ error: "Player ID and action are required" }, { status: 400 })
    }

    const room = roomManager.updatePlayerAction(playerId, action)
    if (!room) {
      return NextResponse.json({ error: "Room or player not found" }, { status: 404 })
    }

    return NextResponse.json({ room })
  } catch (error) {
    console.error("[v0] Error submitting action:", error)
    return NextResponse.json({ error: "Failed to submit action" }, { status: 500 })
  }
}
