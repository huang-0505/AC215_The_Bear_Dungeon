import { type NextRequest, NextResponse } from "next/server"
import { roomManager } from "@/lib/multiplayer/room-manager"
import { openai } from "@ai-sdk/openai"
import { generateText } from "ai"
import { vectorStore } from "@/lib/rag/vector-store"

const campaignSettings = {
  "classic-dungeon": {
    setting:
      "You find yourself at the entrance of an ancient dungeon, carved deep into a mountainside. Stone corridors stretch into darkness ahead.",
    themes: ["combat", "exploration", "treasure hunting"],
    tone: "Classic fantasy adventure with danger lurking around every corner.",
  },
  "wilderness-adventure": {
    setting:
      "You stand at the edge of an untamed wilderness, where ancient forests meet rolling hills under an endless sky.",
    themes: ["survival", "nature", "discovery"],
    tone: "Natural wonder mixed with primal dangers and ancient secrets.",
  },
  "gothic-horror": {
    setting:
      "Mist swirls around a decrepit manor house as thunder rumbles overhead. Something feels deeply wrong about this place.",
    themes: ["horror", "mystery", "supernatural"],
    tone: "Dark and foreboding, with supernatural threats and psychological tension.",
  },
  "political-intrigue": {
    setting:
      "You find yourself in the opulent halls of a noble court, where every smile hides a dagger and every word carries weight.",
    themes: ["roleplay", "investigation", "social"],
    tone: "Sophisticated and tense, where words are weapons and alliances shift like sand.",
  },
  "seafaring-adventure": {
    setting:
      "The salty breeze fills your lungs as you stand on the deck of a ship, endless ocean stretching to the horizon.",
    themes: ["exploration", "naval combat", "piracy"],
    tone: "Adventurous and free-spirited, with the romance and danger of the high seas.",
  },
  "planar-adventure": {
    setting:
      "Reality bends around you as you step through a shimmering portal into a realm where the laws of physics are mere suggestions.",
    themes: ["high magic", "cosmic horror", "multiverse"],
    tone: "Mind-bending and otherworldly, where anything is possible and nothing is certain.",
  },
}

// POST /api/rooms/[roomId]/process-turn - Process all player actions with AI DM
export async function POST(request: NextRequest, { params }: { params: { roomId: string } }) {
  try {
    const room = roomManager.getRoom(params.roomId)
    if (!room) {
      return NextResponse.json({ error: "Room not found" }, { status: 404 })
    }

    if (room.gameState.currentTurn === 0 && room.gameState.messages.length <= room.players.length) {
      // Initialize the adventure with opening scenario
      const campaignData =
        campaignSettings[room.campaign as keyof typeof campaignSettings] || campaignSettings["classic-dungeon"]

      const partyIntro = room.players.map((p) => `${p.name} the ${p.characterClass}`).join(", ")
      const openingMessage = `${campaignData.setting} 

Your party consists of: ${partyIntro}. 

What do you do?`

      roomManager.addMessage(params.roomId, {
        author: "ai",
        text: openingMessage,
      })

      roomManager.startNewTurn(params.roomId)
      return NextResponse.json({ success: true, response: openingMessage })
    }

    // Check if all players are ready
    const allReady = room.players.every((p) => p.isReady)
    if (!allReady) {
      return NextResponse.json({ error: "Not all players are ready" }, { status: 400 })
    }

    // Set room to processing state
    room.gameState.phase = "processing"

    // Collect all player actions
    const playerActions = room.players
      .filter((p) => p.currentAction)
      .map((p) => `${p.name} (${p.characterClass}): ${p.currentAction}`)
      .join("\n")

    console.log("[v0] Processing turn for room:", params.roomId)
    console.log("[v0] Player actions:", playerActions)

    const campaignData =
      campaignSettings[room.campaign as keyof typeof campaignSettings] || campaignSettings["classic-dungeon"]

    // Build context from recent messages
    const recentMessages = room.gameState.messages
      .filter((m) => m.author === "ai" || m.author === "player")
      .slice(-10)
      .map((m) => `${m.author === "ai" ? "DM" : "Player"}: ${m.text}`)
      .join("\n")

    // Try to get RAG context
    let ragContext = ""
    try {
      if (vectorStore.getDocumentCount() > 0) {
        const ragResults = await vectorStore.search(playerActions, 3)
        if (ragResults.length > 0) {
          ragContext =
            "\n\nRELEVANT D&D KNOWLEDGE:\n" +
            ragResults
              .map((result) => `- ${result.chunk.metadata.type.toUpperCase()}: ${result.chunk.content}`)
              .join("\n")
        }
      }
    } catch (error) {
      console.error("[v0] RAG search error:", error)
    }

    const systemPrompt = `You are an expert Dungeon Master for Dungeons & Dragons 5th Edition running a multiplayer campaign. You are running a ${campaignData.tone}

CAMPAIGN: ${room.campaign.replace("-", " ").replace(/\b\w/g, (l) => l.toUpperCase())}
SETTING: ${campaignData.setting}
THEMES: Focus on ${campaignData.themes.join(", ")}

MULTIPLAYER RULES:
- You are managing ${room.players.length} players simultaneously
- Respond to ALL player actions in a cohesive narrative
- Keep responses to 3-4 sentences maximum
- Create consequences that affect the group as a whole
- End with a prompt that allows all players to act again
- Balance spotlight time between all characters

PLAYER PARTY:
${room.players.map((p) => `- ${p.name} (${p.characterClass})`).join("\n")}

CURRENT PLAYER ACTIONS:
${playerActions}

RECENT CONTEXT:
${recentMessages}

${ragContext}

Respond to all player actions as a cohesive group narrative. Address each player's action while weaving them into a single story response.`

    const { text } = await generateText({
      model: openai("gpt-4"),
      prompt: systemPrompt,
      maxOutputTokens: 200,
      temperature: 0.8,
    })

    console.log("[v0] AI DM response:", text)

    // Add player action messages to the room
    for (const player of room.players) {
      if (player.currentAction) {
        roomManager.addMessage(params.roomId, {
          author: "player",
          text: player.currentAction,
          playerId: player.id,
        })
      }
    }

    // Add AI response
    roomManager.addMessage(params.roomId, {
      author: "ai",
      text: text,
    })

    // Start new turn
    roomManager.startNewTurn(params.roomId)

    return NextResponse.json({ success: true, response: text })
  } catch (error) {
    console.error("[v0] Error processing turn:", error)
    return NextResponse.json({ error: "Failed to process turn" }, { status: 500 })
  }
}
