import { NextRequest } from "next/server"
import { proxyJson } from "@/lib/orchestrator"

export async function GET() {
  return proxyJson("/rooms", { method: "GET" })
}

export async function POST(request: NextRequest) {
  const body = await request.text()
  return proxyJson("/rooms", { method: "POST", body })
}
