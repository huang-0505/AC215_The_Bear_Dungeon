import { type NextRequest, NextResponse } from "next/server"
import { roomManager } from "@/lib/multiplayer/room-manager"

// GET /api/rooms - List all rooms
export async function GET() {
  try {
    const rooms = roomManager.listRooms()
    return NextResponse.json({ rooms })
  } catch (error) {
    console.error("[v0] Error listing rooms:", error)
    return NextResponse.json({ error: "Failed to list rooms" }, { status: 500 })
  }
}

// POST /api/rooms - Create a new room
export async function POST(request: NextRequest) {
  try {
    const { name, campaign, creatorId, maxPlayers = 4 } = await request.json()

    if (!name || !campaign || !creatorId) {
      return NextResponse.json({ error: "Name, campaign, and creatorId are required" }, { status: 400 })
    }

    const room = roomManager.createRoom(name, campaign, creatorId, maxPlayers)
    return NextResponse.json({ room })
  } catch (error) {
    console.error("[v0] Error creating room:", error)
    return NextResponse.json({ error: "Failed to create room" }, { status: 500 })
  }
}
