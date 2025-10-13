"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Users, Plus, ArrowLeft, Clock, Trash2 } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import type { GameRoom } from "@/lib/multiplayer/types"

const campaignTypes = [
  { id: "classic-dungeon", name: "Classic Dungeon Crawl" },
  { id: "wilderness-adventure", name: "Wilderness Adventure" },
  { id: "gothic-horror", name: "Gothic Horror" },
  { id: "political-intrigue", name: "Political Intrigue" },
  { id: "seafaring-adventure", name: "Seafaring Adventure" },
  { id: "planar-adventure", name: "Planar Adventure" },
]

export default function MultiplayerLobby() {
  const [rooms, setRooms] = useState<GameRoom[]>([])
  const [loading, setLoading] = useState(true)
  const [showCreateRoom, setShowCreateRoom] = useState(false)
  const [roomName, setRoomName] = useState("")
  const [selectedCampaign, setSelectedCampaign] = useState("")
  const [maxPlayers, setMaxPlayers] = useState("4")
  const [playerName, setPlayerName] = useState("")
  const [playerId, setPlayerId] = useState("")
  const router = useRouter()

  useEffect(() => {
    fetchRooms()
    const savedName = localStorage.getItem("playerName")
    if (savedName) {
      setPlayerName(savedName)
    }
    let savedPlayerId = localStorage.getItem("playerId")
    if (!savedPlayerId) {
      savedPlayerId = Math.random().toString(36).substring(2, 15)
      localStorage.setItem("playerId", savedPlayerId)
    }
    setPlayerId(savedPlayerId)
  }, [])

  const fetchRooms = async () => {
    try {
      const response = await fetch("/api/rooms")
      const data = await response.json()
      setRooms(data.rooms || [])
    } catch (error) {
      console.error("[v0] Error fetching rooms:", error)
    } finally {
      setLoading(false)
    }
  }

  const createRoom = async () => {
    if (!roomName || !selectedCampaign || !playerName) return

    try {
      const response = await fetch("/api/rooms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: roomName,
          campaign: selectedCampaign,
          creatorId: playerId,
          maxPlayers: Number.parseInt(maxPlayers),
        }),
      })

      const data = await response.json()
      if (data.room) {
        localStorage.setItem("playerName", playerName)
        localStorage.setItem("selectedCampaign", selectedCampaign)
        router.push(`/character-select?roomId=${data.room.id}&playerName=${encodeURIComponent(playerName)}`)
      }
    } catch (error) {
      console.error("[v0] Error creating room:", error)
    }
  }

  const deleteRoom = async (roomId: string) => {
    if (!confirm("Are you sure you want to delete this room? This action cannot be undone.")) {
      return
    }

    try {
      const response = await fetch(`/api/rooms/${roomId}/delete`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requesterId: playerId }),
      })

      if (response.ok) {
        fetchRooms()
      } else {
        const error = await response.json()
        alert(error.error || "Failed to delete room")
      }
    } catch (error) {
      console.error("[v0] Error deleting room:", error)
      alert("Failed to delete room")
    }
  }

  const joinRoom = async (roomId: string, campaign: string) => {
    if (!playerName) {
      alert("Please enter your name first")
      return
    }

    localStorage.setItem("playerName", playerName)
    localStorage.setItem("selectedCampaign", campaign)
    router.push(`/character-select?roomId=${roomId}&playerName=${encodeURIComponent(playerName)}`)
  }

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case "waiting":
        return "text-yellow-400 bg-yellow-500/20"
      case "collecting-actions":
        return "text-blue-400 bg-blue-500/20"
      case "processing":
        return "text-purple-400 bg-purple-500/20"
      case "completed":
        return "text-green-400 bg-green-500/20"
      default:
        return "text-gray-400 bg-gray-500/20"
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading multiplayer rooms...</p>
        </div>
      </div>
    )
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
              <Users className="w-3 h-3 mr-1" />
              Multiplayer Adventures
            </Badge>

            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-4 text-glow">Join the Party</h1>

            <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Create or join multiplayer D&D adventures. Collaborate with friends in turn-based gameplay where everyone
              submits their actions before the story continues.
            </p>
          </div>

          {/* Player Name Input */}
          <Card className="mb-8 border-accent/50 bg-accent/10">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <Label htmlFor="playerName" className="text-sm font-medium">
                  Your Name:
                </Label>
                <Input
                  id="playerName"
                  value={playerName}
                  onChange={(e) => setPlayerName(e.target.value)}
                  placeholder="Enter your adventurer name"
                  className="max-w-xs"
                />
              </div>
            </CardContent>
          </Card>

          {/* Create Room Button */}
          <div className="flex justify-between items-center mb-8">
            <h2 className="font-serif text-2xl font-semibold">Available Rooms</h2>

            <Dialog open={showCreateRoom} onOpenChange={setShowCreateRoom}>
              <DialogTrigger asChild>
                <Button className="bg-accent text-accent-foreground hover:bg-accent/90">
                  <Plus className="w-4 h-4 mr-2" />
                  Create Room
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create New Adventure Room</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="roomName">Room Name</Label>
                    <Input
                      id="roomName"
                      value={roomName}
                      onChange={(e) => setRoomName(e.target.value)}
                      placeholder="Epic Adventure Awaits"
                    />
                  </div>

                  <div>
                    <Label htmlFor="campaign">Campaign Type</Label>
                    <Select value={selectedCampaign} onValueChange={setSelectedCampaign}>
                      <SelectTrigger>
                        <SelectValue placeholder="Choose campaign type" />
                      </SelectTrigger>
                      <SelectContent>
                        {campaignTypes.map((campaign) => (
                          <SelectItem key={campaign.id} value={campaign.id}>
                            {campaign.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="maxPlayers">Max Players</Label>
                    <Select value={maxPlayers} onValueChange={setMaxPlayers}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="2">2 Players</SelectItem>
                        <SelectItem value="3">3 Players</SelectItem>
                        <SelectItem value="4">4 Players</SelectItem>
                        <SelectItem value="5">5 Players</SelectItem>
                        <SelectItem value="6">6 Players</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <Button
                    onClick={createRoom}
                    className="w-full"
                    disabled={!roomName || !selectedCampaign || !playerName}
                  >
                    Create & Join Room
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          {/* Rooms List */}
          {rooms.length === 0 ? (
            <Card className="text-center py-12">
              <CardContent>
                <Users className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="font-serif text-xl font-semibold mb-2">No Active Rooms</h3>
                <p className="text-muted-foreground mb-4">
                  Be the first to create an adventure room and invite your friends!
                </p>
                <Button onClick={() => setShowCreateRoom(true)}>
                  <Plus className="w-4 h-4 mr-2" />
                  Create First Room
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {rooms.map((room) => (
                <Card key={room.id} className="hover:border-primary/30 transition-colors">
                  <CardHeader className="pb-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="font-serif text-lg">{room.name}</CardTitle>
                        <p className="text-sm text-muted-foreground mt-1">
                          {campaignTypes.find((c) => c.id === room.campaign)?.name}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className={getPhaseColor(room.gameState.phase)}>
                          {room.gameState.phase.replace("-", " ")}
                        </Badge>
                        {room.creatorId === playerId && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => deleteRoom(room.id)}
                            className="h-6 w-6 p-0 text-red-400 hover:text-red-300 hover:bg-red-500/20"
                          >
                            <Trash2 className="w-3 h-3" />
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="flex items-center text-muted-foreground">
                        <Users className="w-4 h-4 mr-1" />
                        {room.players.length}/{room.maxPlayers} Players
                      </span>
                      <span className="flex items-center text-muted-foreground">
                        <Clock className="w-4 h-4 mr-1" />
                        Turn {room.gameState.currentTurn}
                      </span>
                    </div>

                    {room.players.length > 0 && (
                      <div className="space-y-2">
                        <p className="text-xs text-muted-foreground">Players:</p>
                        <div className="flex flex-wrap gap-1">
                          {room.players.map((player) => (
                            <Badge key={player.id} variant="outline" className="text-xs">
                              {player.name} ({player.characterClass})
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    <Button
                      onClick={() => joinRoom(room.id, room.campaign)}
                      disabled={room.players.length >= room.maxPlayers || !playerName}
                      className="w-full"
                    >
                      {room.players.length >= room.maxPlayers ? "Room Full" : "Join Adventure"}
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
