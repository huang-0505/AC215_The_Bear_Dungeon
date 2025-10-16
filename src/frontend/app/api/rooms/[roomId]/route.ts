import { type NextRequest, NextResponse } from "next/server"
import { roomManager } from "@/lib/multiplayer/room-manager"

// GET /api/rooms/[roomId] - Get room details
export async function GET(request: NextRequest, { params }: { params: { roomId: string } }) {
  try {
    const room = roomManager.getRoom(params.roomId)
    if (!room) {
      return NextResponse.json({ error: "Room not found" }, { status: 404 })
    }

    return NextResponse.json({ room })
  } catch (error) {
    console.error("[v0] Error getting room:", error)
    return NextResponse.json({ error: "Failed to get room" }, { status: 500 })
  }
}
