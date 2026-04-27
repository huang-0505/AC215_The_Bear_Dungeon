"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { ArrowLeft, Plus, Users, Clock } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { Room } from "@/lib/multiplayer/types"

const CHARACTER_CLASSES = [
  "Fighter",
  "Wizard",
  "Ranger",
  "Cleric",
  "Barbarian",
  "Rogue",
] as const

interface RoomsResponse {
  rooms: Room[]
}

interface CreateRoomResponse extends Room {
  player_id: string
  player_token: string
}

interface JoinRoomResponse {
  player_id: string
  player_token: string
  room: Room
}

export default function MultiplayerLobby() {
  const router = useRouter()
  const [rooms, setRooms] = useState<Room[]>([])
  const [loading, setLoading] = useState(true)

  // Player identity persisted to localStorage so the same browser keeps a
  // stable player_id across refreshes and lobby/game transitions.
  const [playerName, setPlayerName] = useState("")
  const [characterName, setCharacterName] = useState("")
  const [characterClass, setCharacterClass] = useState<string>("Fighter")

  // Create-room form state
  const [showCreate, setShowCreate] = useState(false)
  const [roomName, setRoomName] = useState("")
  const [maxPlayers, setMaxPlayers] = useState("4")

  useEffect(() => {
    const savedName = localStorage.getItem("playerName") ?? ""
    const savedCharName = localStorage.getItem("characterName") ?? ""
    const savedClass = localStorage.getItem("characterClass") ?? "Fighter"
    setPlayerName(savedName)
    setCharacterName(savedCharName || savedName)
    setCharacterClass(savedClass)
    void fetchRooms()
  }, [])

  const fetchRooms = async () => {
    setLoading(true)
    try {
      const response = await fetch("/api/rooms")
      if (!response.ok) {
        // eslint-disable-next-line no-console
        console.error("Failed to fetch rooms", response.status)
        setRooms([])
        return
      }
      const data: RoomsResponse = await response.json()
      setRooms(data.rooms ?? [])
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error("Error fetching rooms", error)
    } finally {
      setLoading(false)
    }
  }

  const persistIdentity = () => {
    localStorage.setItem("playerName", playerName)
    localStorage.setItem("characterName", characterName || playerName)
    localStorage.setItem("characterClass", characterClass)
  }

  const buildHostInput = () => ({
    name: playerName,
    character_class: characterClass,
    character_name: characterName || playerName,
  })

  const createRoom = async () => {
    if (!roomName || !playerName) return
    persistIdentity()

    try {
      const response = await fetch("/api/rooms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: roomName,
          host: buildHostInput(),
          max_players: Number.parseInt(maxPlayers, 10),
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        alert(error.detail || "Failed to create room")
        return
      }

      const room: CreateRoomResponse = await response.json()
      const hostPlayer = room.players[0]
      if (!hostPlayer) {
        alert("Room created but host not registered")
        return
      }
      // playerId is durable identity — localStorage. Token is per-room
      // session — sessionStorage so it doesn't outlive the tab.
      localStorage.setItem("playerId", room.player_id)
      sessionStorage.setItem(`playerToken:${room.room_id}`, room.player_token)
      router.push(
        `/multiplayer-game?roomId=${room.room_id}&playerId=${room.player_id}`,
      )
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error("Error creating room", error)
      alert("Failed to create room")
    }
  }

  const joinRoom = async (room: Room) => {
    if (!playerName) {
      alert("Enter your adventurer name first")
      return
    }
    persistIdentity()

    try {
      const response = await fetch(`/api/rooms/${room.room_id}/join`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player: buildHostInput() }),
      })
      if (!response.ok) {
        const error = await response.json()
        alert(error.detail || "Failed to join room")
        return
      }
      const data: JoinRoomResponse = await response.json()
      localStorage.setItem("playerId", data.player_id)
      sessionStorage.setItem(`playerToken:${room.room_id}`, data.player_token)
      router.push(
        `/multiplayer-game?roomId=${room.room_id}&playerId=${data.player_id}`,
      )
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error("Error joining room", error)
      alert("Failed to join room")
    }
  }

  const phaseStyle = (state: Room["state"]) => {
    if (state === "lobby") return "text-yellow-400 bg-yellow-500/20"
    if (state === "active") return "text-blue-400 bg-blue-500/20"
    return "text-gray-400 bg-gray-500/20"
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading multiplayer rooms…</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background text-foreground particle-bg">
      <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-transparent to-transparent" />

      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center magical-glow">
              <div className="w-4 h-4 bg-accent rounded-sm transform rotate-45" />
            </div>
            <span className="font-serif text-xl font-bold text-glow">
              Arcane Engine
            </span>
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
          <div className="text-center mb-12">
            <Badge
              variant="secondary"
              className="mb-6 bg-primary text-primary-foreground border-primary/50"
            >
              <Users className="w-3 h-3 mr-1" />
              Multiplayer Adventures
            </Badge>

            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-4 text-glow">
              Join the Party
            </h1>

            <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Create or join multiplayer D&amp;D adventures. Up to{" "}
              {/* MAX_PARTY_SIZE in orchestrator */}4 players act each round
              before the DM resolves the turn.
            </p>
          </div>

          <Card className="mb-8 border-accent/50 bg-accent/10">
            <CardContent className="p-6 grid sm:grid-cols-3 gap-4">
              <div>
                <Label htmlFor="playerName">Display Name</Label>
                <Input
                  id="playerName"
                  value={playerName}
                  onChange={(e) => setPlayerName(e.target.value)}
                  placeholder="Adventurer"
                />
              </div>
              <div>
                <Label htmlFor="characterName">Character Name</Label>
                <Input
                  id="characterName"
                  value={characterName}
                  onChange={(e) => setCharacterName(e.target.value)}
                  placeholder="Eldrin"
                />
              </div>
              <div>
                <Label htmlFor="characterClass">Class</Label>
                <Select value={characterClass} onValueChange={setCharacterClass}>
                  <SelectTrigger id="characterClass">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {CHARACTER_CLASSES.map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-between items-center mb-8">
            <h2 className="font-serif text-2xl font-semibold">Available Rooms</h2>

            <Dialog open={showCreate} onOpenChange={setShowCreate}>
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
                    <Label htmlFor="maxPlayers">Max Players</Label>
                    <Select value={maxPlayers} onValueChange={setMaxPlayers}>
                      <SelectTrigger id="maxPlayers">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="2">2 Players</SelectItem>
                        <SelectItem value="3">3 Players</SelectItem>
                        <SelectItem value="4">4 Players</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button
                    onClick={createRoom}
                    className="w-full"
                    disabled={!roomName || !playerName || !characterName}
                  >
                    Create &amp; Join Room
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          {rooms.length === 0 ? (
            <Card className="text-center py-12">
              <CardContent>
                <Users className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="font-serif text-xl font-semibold mb-2">
                  No Active Rooms
                </h3>
                <p className="text-muted-foreground mb-4">
                  Be the first to create an adventure room and invite your
                  friends.
                </p>
                <Button onClick={() => setShowCreate(true)}>
                  <Plus className="w-4 h-4 mr-2" />
                  Create First Room
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {rooms.map((room) => (
                <Card
                  key={room.room_id}
                  className="hover:border-primary/30 transition-colors"
                >
                  <CardHeader className="pb-4">
                    <div className="flex items-start justify-between">
                      <CardTitle className="font-serif text-lg">
                        {room.name}
                      </CardTitle>
                      <Badge className={phaseStyle(room.state)}>
                        {room.state.toUpperCase()}
                      </Badge>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="flex items-center text-muted-foreground">
                        <Users className="w-4 h-4 mr-1" />
                        {room.players.length}/{room.max_players} Players
                      </span>
                      <span className="flex items-center text-muted-foreground">
                        <Clock className="w-4 h-4 mr-1" />
                        Round {room.current_round}
                      </span>
                    </div>

                    {room.players.length > 0 && (
                      <div className="space-y-2">
                        <p className="text-xs text-muted-foreground">Players:</p>
                        <div className="flex flex-wrap gap-1">
                          {room.players.map((player) => (
                            <Badge
                              key={player.player_id}
                              variant="outline"
                              className="text-xs"
                            >
                              {player.character_name} ({player.character_class})
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    <Button
                      onClick={() => joinRoom(room)}
                      disabled={
                        room.players.length >= room.max_players ||
                        room.state !== "lobby" ||
                        !playerName
                      }
                      className="w-full"
                    >
                      {room.players.length >= room.max_players
                        ? "Room Full"
                        : room.state !== "lobby"
                          ? "In Progress"
                          : "Join Adventure"}
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
