// Proxy route to orchestrator backend for starting a game
// This route is called as /api/game/start, which nginx forwards to api-gateway:8000/game/start
export async function POST(req: Request) {
  try {
    const body = await req.json()
    const orchestratorUrl = process.env.ORCHESTRATOR_URL || 'http://api-gateway:8000'

    const response = await fetch(`${orchestratorUrl}/game/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `Orchestrator API returned ${response.status}`)
    }

    const data = await response.json()
    return Response.json(data)
  } catch (error: any) {
    console.error('[Orchestrator] Failed to start game:', error)
    return Response.json(
      { error: 'Failed to start game', detail: error?.message },
      { status: 500 }
    )
  }
}
