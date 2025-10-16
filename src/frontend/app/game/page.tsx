"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { DocumentUpload } from "@/components/document-upload"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import Link from "next/link"
import { ArrowLeft, User, BookOpen, Upload } from "lucide-react"

interface Message {
  author: "ai" | "player"
  text: string
  timestamp: number
}

const getCharacterOpening = (characterClass: string, selectedCampaign: string) => {
  const campaignOpenings = {
    "classic-dungeon": {
      Fighter:
        "You are a seasoned warrior, your battle-worn armor gleaming in the torchlight. Your trusty sword rests at your side as you stand before the entrance to an ancient dungeon carved into the mountainside.",
      Wizard:
        "You are a scholar of the arcane arts, your robes rustling with spell components. The magical wards protecting this ancient dungeon call to your very soul.",
      Rogue:
        "You are a master of shadows and stealth, your keen eyes already scanning the dungeon entrance for hidden traps and secret mechanisms.",
      Cleric:
        "You are a devoted servant of the divine, your holy symbol warm against your chest. The evil emanating from this dungeon challenges your faith.",
      Ranger:
        "You are a guardian of the wild places, tracking the monsters that have been emerging from this cursed dungeon.",
      Bard: "You are a keeper of stories and songs, drawn here by tales of the treasures and horrors that lie within these ancient halls.",
    },
    "wilderness-adventure": {
      Fighter:
        "You are a seasoned warrior, your armor adapted for travel through untamed lands. The wilderness stretches endlessly before you.",
      Wizard: "You are a scholar of the arcane arts, studying the natural magic that flows through these wild places.",
      Rogue:
        "You are a master of survival, your skills honed by years of living off the land and avoiding civilization.",
      Cleric: "You are a devoted servant of the divine, spreading your faith to the remote corners of the world.",
      Ranger: "You are truly at home here, one with the wilderness and its creatures.",
      Bard: "You are a wandering storyteller, collecting tales from the far reaches of the world.",
    },
    "gothic-horror": {
      Fighter: "You are a battle-hardened warrior, but even your courage wavers in the face of supernatural dread.",
      Wizard: "You are a scholar of forbidden knowledge, drawn to the dark mysteries that others fear to explore.",
      Rogue: "You are a creature of shadows, but these shadows seem to watch you back with malevolent intent.",
      Cleric: "You are a beacon of divine light in a world growing ever darker with supernatural evil.",
      Ranger: "You are a hunter of monsters, but the creatures here are unlike any you've faced before.",
      Bard: "You are a keeper of stories, but some tales are too terrible to tell.",
    },
    "political-intrigue": {
      Fighter: "You are a warrior sworn to serve, but in these halls, words cut deeper than swords.",
      Wizard:
        "You are a scholar of the arcane arts, using your knowledge to navigate the complex web of court politics.",
      Rogue: "You are a master of secrets and shadows, perfectly suited for the dangerous game of political intrigue.",
      Cleric: "You are a moral compass in a world where ethics are often sacrificed for power.",
      Ranger:
        "You are an outsider to these political games, but your straightforward nature may be exactly what's needed.",
      Bard: "You are perfectly at home in the courts, where charm and wit are your greatest weapons.",
    },
    "seafaring-adventure": {
      Fighter: "You are a warrior of the seas, your sea legs steady beneath you as the ship cuts through the waves.",
      Wizard:
        "You are a scholar of the arcane arts, fascinated by the ancient magics that sleep beneath the ocean's surface.",
      Rogue: "You are quick with both blade and wit, equally at home picking locks or picking pockets in any port.",
      Cleric: "You are a divine beacon for sailors lost in storms, bringing hope to those who brave the endless seas.",
      Ranger: "You are a navigator and scout, reading the signs of wind and wave like others read books.",
      Bard: "You are a teller of sea shanties and sailor's tales, keeping spirits high during long voyages.",
    },
    "planar-adventure": {
      Fighter:
        "You are a warrior whose blade has tasted the essence of multiple realities, hardened by battles across dimensions.",
      Wizard:
        "You are a scholar of the infinite, your mind expanded by exposure to the fundamental forces of the multiverse.",
      Rogue:
        "You are a master of adaptation, your skills honed by surviving in realms where the rules constantly change.",
      Cleric: "You are a divine anchor, maintaining your faith even as reality shifts around you.",
      Ranger: "You are a guide between worlds, helping others navigate the dangerous paths between planes.",
      Bard: "You are a chronicler of impossible tales, your songs weaving magic across multiple dimensions.",
    },
  }

  const campaignData =
    campaignOpenings[selectedCampaign as keyof typeof campaignOpenings] || campaignOpenings["classic-dungeon"]
  return campaignData[characterClass as keyof typeof campaignData] || campaignData.Fighter
}

export default function GameInterface() {
  const [characterClass, setCharacterClass] = useState<string>("")
  const [selectedCampaign, setSelectedCampaign] = useState<string>("")
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState("")
  const [isAiThinking, setIsAiThinking] = useState(false)
  const [showUpload, setShowUpload] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const selectedClass = localStorage.getItem("selectedCharacterClass") || "Fighter"
    const campaign = localStorage.getItem("selectedCampaign") || "classic-dungeon"
    setCharacterClass(selectedClass)
    setSelectedCampaign(campaign)

    const characterOpening = getCharacterOpening(selectedClass, campaign)

    const campaignScenarios = {
      "classic-dungeon":
        "Stone corridors stretch into darkness ahead, and you can hear the distant echo of dripping water. What do you do?",
      "wilderness-adventure":
        "The forest path splits in three directions, each leading deeper into the untamed wilderness. What do you do?",
      "gothic-horror":
        "Lightning illuminates the twisted spires of a decrepit manor house. Something moves in an upstairs window. What do you do?",
      "political-intrigue":
        "Whispered conversations halt as you enter the grand ballroom. All eyes turn to you with calculating interest. What do you do?",
      "seafaring-adventure":
        "The ship's captain points to storm clouds gathering on the horizon. 'We need to make port soon,' he warns. What do you do?",
      "planar-adventure":
        "Reality shimmers around you as you step through the portal. The laws of physics seem... negotiable here. What do you do?",
    }

    const scenario =
      campaignScenarios[campaign as keyof typeof campaignScenarios] || campaignScenarios["classic-dungeon"]

    const initialMessage: Message = {
      author: "ai",
      text: `${characterOpening} ${scenario}`,
      timestamp: Date.now(),
    }
    setMessages([initialMessage])
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || isAiThinking) return

    // Add player message
    const playerMessage: Message = {
      author: "player",
      text: inputValue.trim(),
      timestamp: Date.now(),
    }

    const updatedMessages = [...messages, playerMessage]
    setMessages(updatedMessages)
    setInputValue("")
    setIsAiThinking(true)

    try {
      const response = await fetch("/api/dm", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: updatedMessages,
          characterClass,
          selectedCampaign,
        }),
      })

      const data = await response.json()

      const aiMessage: Message = {
        author: "ai",
        text: data.text,
        timestamp: Date.now(),
      }

      setMessages((prev) => [...prev, aiMessage])
    } catch (error) {
      console.error("Error calling AI DM:", error)
      const errorMessage: Message = {
        author: "ai",
        text: "The mystical connection wavers... *The DM seems to be having trouble hearing you. Please try again.*",
        timestamp: Date.now(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsAiThinking(false)
    }
  }

  const highlightText = (text: string) => {
    // Highlight key words with a golden glow effect
    const keyWords = [
      "door",
      "runes",
      "tree",
      "magic",
      "ancient",
      "stone",
      "warrior",
      "clearing",
      "wizard",
      "rogue",
      "cleric",
      "ranger",
      "bard",
      "fighter",
      "dungeon",
      "forest",
      "manor",
      "court",
      "ship",
      "portal",
      "darkness",
      "light",
      "treasure",
      "danger",
    ]
    let highlightedText = text

    keyWords.forEach((word) => {
      const regex = new RegExp(`\\b${word}\\b`, "gi")
      highlightedText = highlightedText.replace(regex, `<span class="text-amber-300 glow">${word}</span>`)
    })

    return highlightedText
  }

  return (
    <div className="h-screen bg-[#1A1A1A] text-gray-100 font-mono flex flex-col overflow-hidden">
      <div className="absolute top-4 left-4 z-10 flex items-center gap-3">
        <Link href="/story-select">
          <Button
            variant="outline"
            size="sm"
            className="border-accent text-accent hover:bg-accent hover:text-accent-foreground bg-transparent"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Story Select
          </Button>
        </Link>

        {characterClass && (
          <Badge variant="secondary" className="bg-primary text-primary-foreground border-primary/50">
            <User className="w-3 h-3 mr-1" />
            {characterClass}
          </Badge>
        )}

        {selectedCampaign && (
          <Badge variant="outline" className="border-accent text-accent">
            <BookOpen className="w-3 h-3 mr-1" />
            {selectedCampaign.replace("-", " ").replace(/\b\w/g, (l) => l.toUpperCase())}
          </Badge>
        )}
      </div>

      <div className="absolute top-4 right-4 z-10">
        <Dialog open={showUpload} onOpenChange={setShowUpload}>
          <DialogTrigger asChild>
            <Button
              variant="outline"
              size="sm"
              className="border-accent text-accent hover:bg-accent hover:text-accent-foreground bg-transparent"
            >
              <Upload className="w-4 h-4 mr-2" />
              Upload D&D Docs
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Upload D&D Documents</DialogTitle>
            </DialogHeader>
            <DocumentUpload />
          </DialogContent>
        </Dialog>
      </div>

      {/* Narrative Log */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-thin scrollbar-track-gray-800 scrollbar-thumb-gray-600 pt-20">
        {messages.map((message, index) => (
          <div key={index} className="animate-in fade-in duration-300">
            {message.author === "ai" ? (
              <div className="flex gap-3">
                <span className="text-purple-400 font-bold shrink-0">DM:</span>
                <p
                  className="text-gray-100 leading-relaxed"
                  dangerouslySetInnerHTML={{ __html: highlightText(message.text) }}
                />
              </div>
            ) : (
              <div className="flex gap-3">
                <span className="text-cyan-400 font-bold shrink-0">You:</span>
                <p className="text-gray-300 leading-relaxed">{message.text}</p>
              </div>
            )}
          </div>
        ))}

        {isAiThinking && (
          <div className="flex gap-3 animate-in fade-in duration-300">
            <span className="text-purple-400 font-bold shrink-0">DM:</span>
            <div className="flex items-center gap-1">
              <span className="text-gray-400">thinking</span>
              <div className="flex gap-1">
                <div className="w-1 h-1 bg-purple-400 rounded-full animate-pulse"></div>
                <div className="w-1 h-1 bg-purple-400 rounded-full animate-pulse delay-100"></div>
                <div className="w-1 h-1 bg-purple-400 rounded-full animate-pulse delay-200"></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="border-t border-gray-700 bg-[#1A1A1A] p-6">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="What do you do?"
            className="flex-1 bg-gray-800 border-gray-600 text-gray-100 placeholder-gray-400 font-mono focus:border-purple-400 focus:ring-purple-400/20"
            disabled={isAiThinking}
          />
          <Button
            type="submit"
            disabled={!inputValue.trim() || isAiThinking}
            className="bg-purple-600 hover:bg-purple-700 text-white font-mono px-6"
          >
            {isAiThinking ? "..." : ">"}
          </Button>
        </form>

        <p className="text-gray-500 text-xs mt-2 font-mono">
          Press Enter to submit • Arcane Engine v1.0 • Powered by GPT-4 + RAG
        </p>
      </div>

      <style jsx>{`
        .glow {
          text-shadow: 0 0 5px currentColor;
        }
        .scrollbar-thin::-webkit-scrollbar {
          width: 6px;
        }
        .scrollbar-track-gray-800::-webkit-scrollbar-track {
          background: #374151;
        }
        .scrollbar-thumb-gray-600::-webkit-scrollbar-thumb {
          background: #6B7280;
          border-radius: 3px;
        }
        .scrollbar-thumb-gray-600::-webkit-scrollbar-thumb:hover {
          background: #9CA3AF;
        }
      `}</style>
    </div>
  )
}
