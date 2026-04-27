import { orchestratorUrl } from "@/lib/orchestrator"

// Stream the orchestrator's SSE response straight through to the client.
// Next.js Edge runtime is not used because we need to support an indefinite
// stream from a Node.js fetch.
export const dynamic = "force-dynamic"

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export async function GET(
  request: Request,
  { params }: { params: { roomId: string } },
) {
  if (!UUID_RE.test(params.roomId)) {
    return new Response(JSON.stringify({ error: "Invalid room ID" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    })
  }

  // Forward the client's AbortSignal upstream so closing the browser SSE
  // connection cancels the in-flight fetch to the orchestrator instead of
  // leaking the upstream TCP connection.
  const upstream = await fetch(orchestratorUrl(`/rooms/${params.roomId}/events`), {
    method: "GET",
    headers: { Accept: "text/event-stream" },
    cache: "no-store",
    signal: request.signal,
  })

  if (!upstream.ok || !upstream.body) {
    return new Response(
      JSON.stringify({ error: "Failed to connect to room event stream" }),
      { status: upstream.status || 502, headers: { "Content-Type": "application/json" } },
    )
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  })
}
