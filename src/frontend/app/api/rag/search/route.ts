import { type NextRequest, NextResponse } from "next/server"
import { vectorStore } from "@/lib/rag/vector-store"

export async function POST(request: NextRequest) {
  try {
    const { query, type, limit = 5 } = await request.json()

    if (!query || typeof query !== "string") {
      return NextResponse.json({ error: "Query is required" }, { status: 400 })
    }

    const results = type ? await vectorStore.searchByType(query, type, limit) : await vectorStore.search(query, limit)

    return NextResponse.json({
      results: results.map((r) => ({
        content: r.chunk.content,
        metadata: r.chunk.metadata,
        similarity: r.similarity,
      })),
      count: results.length,
    })
  } catch (error) {
    console.error("[v0] Error searching documents:", error)
    return NextResponse.json({ error: "Failed to search documents" }, { status: 500 })
  }
}
