"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Sword, Shield, Sparkles, BookOpen, Crosshair, Music, ArrowRight, ArrowLeft, Users } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"

const characterClasses = [
  {
    name: "Fighter",
    icon: Sword,
    description: "Masters of martial combat, skilled with a variety of weapons and armor.",
    traits: ["High Health", "Weapon Mastery", "Combat Tactics"],
    color: "text-red-400",
    bgColor: "bg-red-500/20",
    borderColor: "border-red-500/30",
  },
  {
    name: "Wizard",
    icon: Sparkles,
    description: "Scholarly magic-users capable of manipulating the structures of reality.",
    traits: ["Powerful Spells", "Ritual Magic", "Arcane Knowledge"],
    color: "text-blue-400",
    bgColor: "bg-blue-500/20",
    borderColor: "border-blue-500/30",
  },
  {
    name: "Rogue",
    icon: Crosshair,
    description: "Skilled in stealth and precision, masters of locks, traps, and sneak attacks.",
    traits: ["Stealth Expert", "Sneak Attack", "Skill Versatility"],
    color: "text-purple-400",
    bgColor: "bg-purple-500/20",
    borderColor: "border-purple-500/30",
  },
  {
    name: "Cleric",
    icon: Shield,
    description: "Divine spellcasters who serve the gods and heal their allies.",
    traits: ["Divine Magic", "Healing Powers", "Turn Undead"],
    color: "text-yellow-400",
    bgColor: "bg-yellow-500/20",
    borderColor: "border-yellow-500/30",
  },
  {
    name: "Ranger",
    icon: BookOpen,
    description: "Warriors of the wilderness, skilled in tracking, survival, and combat.",
    traits: ["Nature Magic", "Tracking", "Favored Enemy"],
    color: "text-green-400",
    bgColor: "bg-green-500/20",
    borderColor: "border-green-500/30",
  },
  {
    name: "Bard",
    icon: Music,
    description: "Masters of song, speech, and the magic they contain.",
    traits: ["Inspiration", "Versatile Magic", "Social Skills"],
    color: "text-pink-400",
    bgColor: "bg-pink-500/20",
    borderColor: "border-pink-500/30",
  },
]

export default function CharacterSelect() {
  const [selectedClass, setSelectedClass] = useState<string | null>(null)
  const [isMultiplayer, setIsMultiplayer] = useState(false)
  const [roomId, setRoomId] = useState<string | null>(null)
  const [playerName, setPlayerName] = useState<string>("")
  const router = useRouter()

  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search)
    const roomIdParam = urlParams.get("roomId")
    const playerNameParam = urlParams.get("playerName")

    if (roomIdParam && playerNameParam) {
      // This is multiplayer mode - store in localStorage for the game
      setIsMultiplayer(true)
      setRoomId(roomIdParam)
      setPlayerName(playerNameParam)
      localStorage.setItem("roomId", roomIdParam)
      localStorage.setItem("playerName", playerNameParam)
    } else {
      // This is solo mode - clear any leftover multiplayer data
      setIsMultiplayer(false)
      localStorage.removeItem("roomId")
      localStorage.removeItem("playerName")
      localStorage.removeItem("playerId")
    }
  }, [])

  const handleStartAdventure = async () => {
    if (!selectedClass) return

    // Store selected class in localStorage for the game to use
    localStorage.setItem("selectedCharacterClass", selectedClass)

    if (isMultiplayer && roomId) {
      // Join the multiplayer room
      try {
        const playerId = `${playerName}-${Date.now()}`
        const response = await fetch(`/api/rooms/${roomId}/join`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            playerId,
            playerName,
            characterClass: selectedClass,
          }),
        })

        if (response.ok) {
          // Store player ID for the game
          localStorage.setItem("playerId", playerId)
          router.push("/multiplayer-game")
        } else {
          const error = await response.json()
          alert(error.error || "Failed to join room")
        }
      } catch (error) {
        console.error("[v0] Error joining room:", error)
        alert("Failed to join room")
      }
    } else {
      // Single player mode
      router.push("/game")
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground particle-bg">
      <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-transparent to-transparent"></div>

      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center magical-glow">
              <div className="w-4 h-4 bg-accent rounded-sm transform rotate-45"></div>
            </div>
            <span className="font-serif text-xl font-bold text-glow">Arcane Engine</span>
          </Link>

          <Link href={isMultiplayer ? "/multiplayer" : "/"}>
            <Button
              variant="outline"
              size="sm"
              className="border-accent text-foreground hover:bg-accent hover:text-accent-foreground bg-transparent"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              {isMultiplayer ? "Back to Multiplayer" : "Back to Home"}
            </Button>
          </Link>
        </div>
      </header>

      <div className="container relative py-12">
        <div className="max-w-4xl mx-auto">
          {/* Title Section */}
          <div className="text-center mb-12">
            <Badge variant="secondary" className="mb-6 bg-primary text-primary-foreground border-primary/50">
              <Sparkles className="w-3 h-3 mr-1" />
              Character Creation
              {isMultiplayer && (
                <>
                  <Users className="w-3 h-3 ml-2 mr-1" />
                  Multiplayer
                </>
              )}
            </Badge>

            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-4 text-glow">Choose Your Champion</h1>

            <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              {isMultiplayer
                ? `Select your character class to join the multiplayer adventure as ${playerName}.`
                : "Select your character class to begin your epic adventure. Each class offers unique abilities and playstyles."}
            </p>
          </div>

          {/* Character Classes Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
            {characterClasses.map((characterClass) => {
              const Icon = characterClass.icon
              const isSelected = selectedClass === characterClass.name

              return (
                <Card
                  key={characterClass.name}
                  className={`cursor-pointer transition-all duration-300 hover:scale-105 ${
                    isSelected
                      ? `${characterClass.borderColor} ${characterClass.bgColor} magical-glow`
                      : "border-border hover:border-primary/30"
                  }`}
                  onClick={() => setSelectedClass(characterClass.name)}
                >
                  <CardHeader className="text-center pb-4">
                    <div
                      className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 ${
                        isSelected ? characterClass.bgColor : "bg-primary/20"
                      }`}
                    >
                      <Icon className={`w-8 h-8 ${isSelected ? characterClass.color : "text-primary"}`} />
                    </div>
                    <CardTitle className="font-serif text-xl">{characterClass.name}</CardTitle>
                  </CardHeader>

                  <CardContent className="pt-0">
                    <p className="text-sm text-muted-foreground mb-4 text-center">{characterClass.description}</p>

                    <div className="space-y-2">
                      {characterClass.traits.map((trait, index) => (
                        <div key={index} className="flex items-center text-sm">
                          <div
                            className={`w-2 h-2 rounded-full mr-2 ${
                              isSelected ? characterClass.color.replace("text-", "bg-") : "bg-accent"
                            }`}
                          ></div>
                          <span className={isSelected ? characterClass.color : "text-foreground"}>{trait}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>

          {/* Selected Character Info */}
          {selectedClass && (
            <Card className="mb-8 border-accent/50 bg-accent/10">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-serif text-xl font-semibold mb-2">
                      Ready to adventure as a {selectedClass}?
                      {isMultiplayer && <span className="text-accent ml-2">({playerName})</span>}
                    </h3>
                    <p className="text-muted-foreground">
                      {isMultiplayer
                        ? "Join your party and start the multiplayer adventure!"
                        : "Your character will be fully customized during the adventure based on your choices and actions."}
                    </p>
                  </div>
                  <Button
                    size="lg"
                    onClick={handleStartAdventure}
                    className="bg-accent text-accent-foreground hover:bg-accent/90 gold-glow"
                  >
                    {isMultiplayer ? "Join Adventure" : "Begin Adventure"}
                    <ArrowRight className="w-5 h-5 ml-2" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Instructions */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground">
              {isMultiplayer
                ? "Once you join, you'll wait for other players to be ready before the adventure begins."
                : "Don't worry about the details yet - our AI will help you develop your character's personality, backstory, and specific abilities as you play."}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
