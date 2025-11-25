// Proxy route to orchestrator for submitting combat actions (orchestrator proxies to combat agent)
export async function POST(
  req: Request,
  { params }: { params: { sessionId: string } }
) {
  try {
    const { sessionId } = params
    const body = await req.json()
    // Route through orchestrator (api-gateway) which has access to combat-agent
    const orchestratorUrl = process.env.ORCHESTRATOR_URL || 'http://api-gateway:8000'

    const response = await fetch(`${orchestratorUrl}/combat/action/${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      // Add timeout to prevent indefinite hanging (15 seconds to account for GenAI processing)
      signal: AbortSignal.timeout(15000), // 15 second timeout
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `Orchestrator API returned ${response.status}`)
    }

    const data = await response.json()
    return Response.json(data)
  } catch (error: any) {
    console.error('[Combat Agent] Failed to submit combat action:', error)
    return Response.json(
      { error: 'Failed to submit combat action', detail: error?.message },
      { status: 500 }
    )
  }
}

