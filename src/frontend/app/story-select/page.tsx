"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Castle, TreePine, Skull, Crown, Compass, Zap, ArrowRight, ArrowLeft, Users, Clock, Star } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"

const campaignTypes = [
  {
    id: "classic-dungeon",
    name: "Classic Dungeon Crawl",
    icon: Castle,
    description: "Explore ancient dungeons filled with monsters, traps, and treasure.",
    setting: "Underground ruins and forgotten temples",
    difficulty: "Beginner Friendly",
    duration: "2-4 hours",
    themes: ["Combat", "Exploration", "Treasure Hunting"],
    color: "text-orange-400",
    bgColor: "bg-orange-500/20",
    borderColor: "border-orange-500/30",
  },
  {
    id: "wilderness-adventure",
    name: "Wilderness Adventure",
    icon: TreePine,
    description: "Journey through untamed lands, encounter wild beasts and ancient secrets.",
    setting: "Forests, mountains, and wild frontiers",
    difficulty: "Moderate",
    duration: "3-5 hours",
    themes: ["Survival", "Nature", "Discovery"],
    color: "text-green-400",
    bgColor: "bg-green-500/20",
    borderColor: "border-green-500/30",
  },
  {
    id: "gothic-horror",
    name: "Gothic Horror",
    icon: Skull,
    description: "Face supernatural terrors in a world of darkness and dread.",
    setting: "Haunted castles, cursed villages, and shadowy realms",
    difficulty: "Advanced",
    duration: "4-6 hours",
    themes: ["Horror", "Mystery", "Supernatural"],
    color: "text-purple-400",
    bgColor: "bg-purple-500/20",
    borderColor: "border-purple-500/30",
  },
  {
    id: "political-intrigue",
    name: "Political Intrigue",
    icon: Crown,
    description: "Navigate court politics, espionage, and power struggles.",
    setting: "Royal courts, noble houses, and city-states",
    difficulty: "Advanced",
    duration: "3-5 hours",
    themes: ["Roleplay", "Investigation", "Social"],
    color: "text-blue-400",
    bgColor: "bg-blue-500/20",
    borderColor: "border-blue-500/30",
  },
  {
    id: "seafaring-adventure",
    name: "Seafaring Adventure",
    icon: Compass,
    description: "Sail the high seas in search of treasure and adventure.",
    setting: "Ships, islands, and coastal towns",
    difficulty: "Moderate",
    duration: "3-4 hours",
    themes: ["Exploration", "Naval Combat", "Piracy"],
    color: "text-cyan-400",
    bgColor: "bg-cyan-500/20",
    borderColor: "border-cyan-500/30",
  },
  {
    id: "planar-adventure",
    name: "Planar Adventure",
    icon: Zap,
    description: "Travel between dimensions and face cosmic threats.",
    setting: "Elemental planes, celestial realms, and otherworldly domains",
    difficulty: "Expert",
    duration: "5-8 hours",
    themes: ["High Magic", "Cosmic Horror", "Multiverse"],
    color: "text-pink-400",
    bgColor: "bg-pink-500/20",
    borderColor: "border-pink-500/30",
  },
]

const difficultyColors = {
  "Beginner Friendly": "text-green-400 bg-green-500/20",
  Moderate: "text-yellow-400 bg-yellow-500/20",
  Advanced: "text-orange-400 bg-orange-500/20",
  Expert: "text-red-400 bg-red-500/20",
}

export default function StorySelect() {
  const [selectedCampaign, setSelectedCampaign] = useState<string | null>(null)
  const router = useRouter()

  const handleContinue = () => {
    if (selectedCampaign) {
      // Store selected campaign in localStorage
      localStorage.setItem("selectedCampaign", selectedCampaign)
      router.push("/character-select")
    }
  }

  const selectedCampaignData = campaignTypes.find((c) => c.id === selectedCampaign)

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

          <Link href="/">
            <Button
              variant="outline"
              size="sm"
              className="border-accent text-foreground hover:bg-accent hover:text-accent-foreground bg-transparent"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </Button>
          </Link>
        </div>
      </header>

      <div className="container relative py-12">
        <div className="max-w-6xl mx-auto">
          {/* Title Section */}
          <div className="text-center mb-12">
            <Badge variant="secondary" className="mb-6 bg-primary text-primary-foreground border-primary/50">
              <Star className="w-3 h-3 mr-1" />
              Campaign Selection
            </Badge>

            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-4 text-glow">Choose Your Adventure</h1>

            <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Select the type of campaign you'd like to experience. Each offers unique challenges, settings, and
              storytelling opportunities powered by our AI Dungeon Master.
            </p>
          </div>

          {/* Campaign Types Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
            {campaignTypes.map((campaign) => {
              const Icon = campaign.icon
              const isSelected = selectedCampaign === campaign.id

              return (
                <Card
                  key={campaign.id}
                  className={`cursor-pointer transition-all duration-300 hover:scale-105 ${
                    isSelected
                      ? `${campaign.borderColor} ${campaign.bgColor} magical-glow`
                      : "border-border hover:border-primary/30"
                  }`}
                  onClick={() => setSelectedCampaign(campaign.id)}
                >
                  <CardHeader className="pb-4">
                    <div className="flex items-start justify-between mb-4">
                      <div
                        className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                          isSelected ? campaign.bgColor : "bg-primary/20"
                        }`}
                      >
                        <Icon className={`w-6 h-6 ${isSelected ? campaign.color : "text-primary"}`} />
                      </div>
                      <Badge
                        variant="secondary"
                        className={`text-xs ${difficultyColors[campaign.difficulty as keyof typeof difficultyColors]}`}
                      >
                        {campaign.difficulty}
                      </Badge>
                    </div>
                    <CardTitle className="font-serif text-lg leading-tight">{campaign.name}</CardTitle>
                  </CardHeader>

                  <CardContent className="pt-0 space-y-4">
                    <p className="text-sm text-muted-foreground leading-relaxed">{campaign.description}</p>

                    <div className="space-y-3">
                      <div className="flex items-center text-xs text-muted-foreground">
                        <Castle className="w-3 h-3 mr-2" />
                        {campaign.setting}
                      </div>

                      <div className="flex items-center text-xs text-muted-foreground">
                        <Clock className="w-3 h-3 mr-2" />
                        {campaign.duration}
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-1">
                      {campaign.themes.map((theme, index) => (
                        <Badge
                          key={index}
                          variant="outline"
                          className={`text-xs ${
                            isSelected ? `${campaign.color} border-current` : "text-muted-foreground"
                          }`}
                        >
                          {theme}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>

          {/* Selected Campaign Details */}
          {selectedCampaignData && (
            <Card className="mb-8 border-accent/50 bg-accent/10">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h3 className="font-serif text-xl font-semibold mb-2 flex items-center">
                      <selectedCampaignData.icon className={`w-5 h-5 mr-2 ${selectedCampaignData.color}`} />
                      {selectedCampaignData.name} Selected
                    </h3>
                    <p className="text-muted-foreground mb-3">{selectedCampaignData.description}</p>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <span className="flex items-center">
                        <Users className="w-4 h-4 mr-1" />
                        Solo Adventure
                      </span>
                      <span className="flex items-center">
                        <Clock className="w-4 h-4 mr-1" />
                        {selectedCampaignData.duration}
                      </span>
                    </div>
                  </div>
                  <Button
                    size="lg"
                    onClick={handleContinue}
                    className="bg-accent text-accent-foreground hover:bg-accent/90 gold-glow ml-6"
                  >
                    Continue to Character Creation
                    <ArrowRight className="w-5 h-5 ml-2" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Instructions */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground max-w-2xl mx-auto">
              Your selected campaign will influence the AI Dungeon Master's storytelling style, the types of encounters
              you'll face, and the overall tone of your adventure.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
