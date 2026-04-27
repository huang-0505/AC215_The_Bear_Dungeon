import { proxyJson } from "@/lib/orchestrator"

export async function GET(
  _request: Request,
  { params }: { params: { roomId: string } },
) {
  return proxyJson(`/rooms/${params.roomId}`, { method: "GET" })
}
