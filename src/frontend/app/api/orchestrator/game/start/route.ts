// Proxy route to orchestrator backend for starting a game
export async function POST(req: Request) {
  try {
    const body = await req.json()
    const orchestratorUrl = 'http://api-gateway:8000'

    const response = await fetch(`${orchestratorUrl}/game/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `Orchestrator API returned ${response.status}`)
    }

    const data = await response.json()
    return Response.json(data)
  } catch (error) {
    console.error('[Orchestrator] Failed to start game:', error)
    return Response.json(
      { error: 'Failed to start game' },
      { status: 500 }
    )
  }
}
