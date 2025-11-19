// Proxy route to orchestrator backend for campaigns
export async function GET() {
  try {
    // In Docker, use internal service name; otherwise use environment variable
    const orchestratorUrl = 'http://api-gateway:8000'
    const response = await fetch(`${orchestratorUrl}/campaigns`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`Orchestrator API returned ${response.status}`)
    }

    const data = await response.json()
    return Response.json(data)
  } catch (error) {
    console.error('[Orchestrator] Failed to fetch campaigns:', error)
    return Response.json(
      { error: 'Failed to fetch campaigns from orchestrator' },
      { status: 500 }
    )
  }
}
