import { proxyJson } from "@/lib/orchestrator"

export async function GET(
  _request: Request,
  { params }: { params: { sessionId: string } },
) {
  return proxyJson(`/game/state/${params.sessionId}`, { method: "GET" })
}
