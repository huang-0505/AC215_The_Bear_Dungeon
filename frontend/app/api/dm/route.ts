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

export async function POST(req: Request) {
  const { messages, characterClass, selectedCampaign } = await req.json()

  console.log("[v0] API Key exists:", !!process.env.OPENAI_API_KEY)
  console.log("[v0] Character class:", characterClass)
  console.log("[v0] Selected campaign:", selectedCampaign)
  console.log("[v0] Messages count:", messages.length)

  const campaignData =
    campaignSettings[selectedCampaign as keyof typeof campaignSettings] || campaignSettings["classic-dungeon"]

  const latestPlayerMessage = messages.filter((msg: any) => msg.author === "player").pop()?.text || ""

  let ragContext = ""
  try {
    if (latestPlayerMessage && vectorStore.getDocumentCount() > 0) {
      console.log("[v0] Searching RAG for:", latestPlayerMessage)
      const ragResults = await vectorStore.search(latestPlayerMessage, 3)

      if (ragResults.length > 0) {
        ragContext =
          "\n\nRELEVANT D&D KNOWLEDGE:\n" +
          ragResults
            .map((result) => `- ${result.chunk.metadata.type.toUpperCase()}: ${result.chunk.content}`)
            .join("\n")
        console.log("[v0] RAG context found:", ragResults.length, "results")
      }
    }
  } catch (error) {
    console.error("[v0] RAG search error:", error)
    // Continue without RAG if it fails
  }

  const systemPrompt = `You are an expert Dungeon Master for Dungeons & Dragons 5th Edition. You are running a ${campaignData.tone}

CAMPAIGN: ${selectedCampaign?.replace("-", " ").replace(/\b\w/g, (l) => l.toUpperCase()) || "Classic Adventure"}
SETTING: ${campaignData.setting}
THEMES: Focus on ${campaignData.themes.join(", ")}

PLAYER CHARACTER: ${characterClass}

RULES:
- Keep responses to 2-3 sentences maximum
- Be descriptive but concise
- Always end with a subtle prompt for the player's next action
- Maintain the campaign's tone and themes
- Use D&D 5e rules and terminology accurately
- Adapt responses based on the player's character class and campaign type
- Create engaging, interactive scenarios appropriate to the campaign
- Don't reveal solutions too easily

CHARACTER CLASS CONTEXT:
- Fighter: Focus on combat opportunities, physical challenges, tactical situations
- Wizard: Emphasize magical elements, arcane mysteries, spell-casting opportunities  
- Rogue: Highlight stealth options, hidden mechanisms, traps, secret passages
- Cleric: Include divine elements, holy/unholy presences, healing opportunities
- Ranger: Connect with nature, animal companions, tracking, wilderness survival
- Bard: Social interactions, lore, music/performance magic, inspiring others

${ragContext}

Previous conversation:
${messages.map((msg: any) => `${msg.author === "player" ? "Player" : "DM"}: ${msg.text}`).join("\n")}`

  try {
    console.log("[v0] Making OpenAI API call...")

    const { text } = await generateText({
      model: openai("gpt-4"),
      prompt: systemPrompt,
      maxOutputTokens: 150,
      temperature: 0.8,
    })

    console.log("[v0] OpenAI response received:", text.substring(0, 100) + "...")

    return Response.json({ text })
  } catch (error) {
    console.error("[v0] AI DM Error details:", error)
    console.error("[v0] Error message:", error instanceof Error ? error.message : "Unknown error")

    return Response.json(
      {
        text: "The ancient magic flickers momentarily, but the DM's voice returns: 'Something went wrong with the mystical connection. Please try again.'",
      },
      { status: 500 },
    )
  }
}
