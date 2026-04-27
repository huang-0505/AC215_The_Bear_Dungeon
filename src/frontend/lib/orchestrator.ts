// Centralized helper for proxying to the orchestrator.

const ORCHESTRATOR_URL =
  process.env.ORCHESTRATOR_URL || "http://api-gateway:8000"

// Headers we forward upstream when present on the incoming request. Token is
// the only auth signal today; add others (Authorization, etc.) here later.
const FORWARDED_HEADERS = ["x-player-token"] as const

export function orchestratorUrl(path: string): string {
  return `${ORCHESTRATOR_URL}${path}`
}

/**
 * Proxy a Next.js Route Handler request to the orchestrator. Pass the
 * incoming Request when the call needs to forward auth headers; omit it
 * for public reads (e.g. GET /rooms).
 */
export async function proxyJson(
  path: string,
  init: RequestInit = {},
  incoming?: Request,
): Promise<Response> {
  const headers = new Headers(init.headers)
  if (!headers.has("Content-Type") && init.body) {
    headers.set("Content-Type", "application/json")
  }
  if (incoming) {
    for (const name of FORWARDED_HEADERS) {
      const value = incoming.headers.get(name)
      if (value) headers.set(name, value)
    }
  }
  const response = await fetch(orchestratorUrl(path), { ...init, headers })
  const text = await response.text()
  return new Response(text, {
    status: response.status,
    headers: { "Content-Type": "application/json" },
  })
}
