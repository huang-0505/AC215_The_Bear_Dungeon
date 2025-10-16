import { type NextRequest, NextResponse } from "next/server"
import { DocumentProcessor } from "@/lib/rag/document-processor"
import { vectorStore } from "@/lib/rag/vector-store"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File
    const type = formData.get("type") as string
    const title = formData.get("title") as string

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    if (!type || !["rule", "lore", "monster", "spell", "item", "campaign"].includes(type)) {
      return NextResponse.json({ error: "Invalid document type" }, { status: 400 })
    }

    let chunks: any[] = []

    // Handle text files
    if (file.type === "text/plain") {
      const text = await file.text()
      chunks = await DocumentProcessor.processText(text, {
        source: file.name,
        type: type as any,
        title: title || file.name,
      })
    } else if (file.type === "application/pdf") {
      chunks = await DocumentProcessor.processPDF(file, {
        type: type as any,
        title: title || file.name,
      })
    } else {
      return NextResponse.json({ error: "Unsupported file type. Please upload PDF or text files." }, { status: 400 })
    }

    await vectorStore.addDocuments(chunks)

    return NextResponse.json({
      success: true,
      message: `Processed ${chunks.length} chunks from ${file.name}`,
      chunksCount: chunks.length,
      documentCount: vectorStore.getDocumentCount(),
    })
  } catch (error) {
    console.error("[v0] Error processing document:", error)
    return NextResponse.json({ error: "Failed to process document" }, { status: 500 })
  }
}
