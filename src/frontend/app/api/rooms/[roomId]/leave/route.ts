import { NextRequest } from "next/server"
import { proxyJson } from "@/lib/orchestrator"

export async function POST(
  request: NextRequest,
  { params }: { params: { roomId: string } },
) {
  const body = await request.text()
  return proxyJson(`/rooms/${params.roomId}/leave`, { method: "POST", body }, request)
}
