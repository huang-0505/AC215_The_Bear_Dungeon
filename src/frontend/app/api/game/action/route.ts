// Proxy route to orchestrator backend for game actions
// This route is called as /api/game/action, which nginx forwards to api-gateway:8000/game/action
export async function POST(req: Request) {
  try {
    const body = await req.json()
    const orchestratorUrl = process.env.ORCHESTRATOR_URL || 'http://api-gateway:8000'

    const response = await fetch(`${orchestratorUrl}/game/action`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: body.session_id,
        text: body.text,
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `Orchestrator API returned ${response.status}`)
    }

    const data = await response.json()
    return Response.json(data)
  } catch (error: any) {
    console.error('[Orchestrator] Failed to process action:', error)
    return Response.json(
      { error: 'Failed to process action', detail: error?.message },
      { status: 500 }
    )
  }
}
