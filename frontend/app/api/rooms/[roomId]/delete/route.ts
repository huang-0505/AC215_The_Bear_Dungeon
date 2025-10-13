import { type NextRequest, NextResponse } from "next/server"
import { roomManager } from "@/lib/multiplayer/room-manager"

// DELETE /api/rooms/[roomId]/delete - Delete a room (creator only)
export async function DELETE(request: NextRequest, { params }: { params: { roomId: string } }) {
  try {
    const { requesterId } = await request.json()

    if (!requesterId) {
      return NextResponse.json({ error: "Requester ID is required" }, { status: 400 })
    }

    const success = roomManager.deleteRoom(params.roomId, requesterId)

    if (!success) {
      return NextResponse.json({ error: "Room not found" }, { status: 404 })
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("[v0] Error deleting room:", error)

    if (error instanceof Error && error.message.includes("Only the room creator")) {
      return NextResponse.json({ error: error.message }, { status: 403 })
    }

    return NextResponse.json({ error: "Failed to delete room" }, { status: 500 })
  }
}
