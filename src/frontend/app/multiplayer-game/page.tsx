"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { useRoomEvents } from "@/hooks/use-room-events"
import { Users, Clock, ArrowLeft, CheckCircle, Circle } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"

export default function MultiplayerGame() {
  const [roomId, setRoomId] = useState<string | null>(null)
  const [playerId, setPlayerId] = useState<string>("")
  const [playerName, setPlayerName] = useState<string>("")
  const [characterClass, setCharacterClass] = useState<string>("")
  const [inputValue, setInputValue] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const { room, isConnected } = useRoomEvents(roomId)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const router = useRouter()

  useEffect(() => {
    // Get room and player info from localStorage
    const storedRoomId = localStorage.getItem("roomId")
    const storedPlayerName = localStorage.getItem("playerName")
    const storedCharacterClass = localStorage.getItem("selectedCharacterClass")
    const storedPlayerId = localStorage.getItem("playerId")

    if (!storedRoomId || !storedPlayerName || !storedCharacterClass) {
      router.push("/multiplayer")
      return
    }

    setRoomId(storedRoomId)
    setPlayerName(storedPlayerName)
    setCharacterClass(storedCharacterClass)
    setPlayerId(storedPlayerId || `${storedPlayerName}-${Date.now()}`)
  }, [router])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [room?.gameState.messages])

  useEffect(() => {
    if (room?.gameState.phase === "processing" && !isProcessing) {
      processTurn()
    }
  }, [room?.gameState.phase])

  useEffect(() => {
    if (room && room.gameState.messages.length === 1 && room.gameState.currentTurn === 0) {
      // This is a new room, start the first turn
      const startFirstTurn = async () => {
        try {
          await fetch(`/api/rooms/${roomId}/process-turn`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
          })
        } catch (error) {
          console.error("[v0] Error starting first turn:", error)
        }
      }

      if (room.players.length > 0) {
        startFirstTurn()
      }
    }
  }, [room, roomId])

  const submitAction = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || !roomId || !playerId || isSubmitting) return

    setIsSubmitting(true)
    try {
      const response = await fetch(`/api/rooms/${roomId}/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          playerId,
          action: inputValue.trim(),
        }),
      })

      if (response.ok) {
        setInputValue("")
      }
    } catch (error) {
      console.error("[v0] Error submitting action:", error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const processTurn = async () => {
    if (!roomId || isProcessing) return

    setIsProcessing(true)
    try {
      const response = await fetch(`/api/rooms/${roomId}/process-turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      })

      if (!response.ok) {
        console.error("[v0] Error processing turn:", await response.text())
      }
    } catch (error) {
      console.error("[v0] Error processing turn:", error)
    } finally {
      setIsProcessing(false)
    }
  }

  const currentPlayer = room?.players.find((p) => p.id === playerId)
  const allPlayersReady = room?.players.every((p) => p.isReady) || false
  const waitingForPlayers = room?.players.filter((p) => !p.isReady) || []

  if (!room) {
    return (
      <div className="h-screen bg-[#1A1A1A] text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-purple-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">{isConnected ? "Loading room..." : "Connecting to room..."}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen bg-[#1A1A1A] text-gray-100 font-mono flex flex-col overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-700 bg-[#1A1A1A] p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/multiplayer">
              <Button
                variant="outline"
                size="sm"
                className="border-accent text-accent hover:bg-accent hover:text-accent-foreground bg-transparent"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Leave Room
              </Button>
            </Link>

            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="bg-primary text-primary-foreground">
                <Users className="w-3 h-3 mr-1" />
                {room.name}
              </Badge>

              <Badge variant="outline" className="border-accent text-accent">
                <Clock className="w-3 h-3 mr-1" />
                Turn {room.gameState.currentTurn}
              </Badge>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-400" : "bg-red-400"}`}></div>
            <span className="text-xs text-gray-400">{isConnected ? "Connected" : "Disconnected"}</span>
          </div>
        </div>
      </div>

      {/* Players Panel */}
      <div className="border-b border-gray-700 bg-gray-800/50 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-400">Players:</span>
            <div className="flex gap-2">
              {room.players.map((player) => (
                <Badge
                  key={player.id}
                  variant={player.id === playerId ? "default" : "outline"}
                  className={`text-xs flex items-center gap-1 ${
                    player.isReady ? "border-green-400 text-green-400" : "border-gray-500 text-gray-400"
                  }`}
                >
                  {player.isReady ? <CheckCircle className="w-3 h-3" /> : <Circle className="w-3 h-3" />}
                  {player.name} ({player.characterClass})
                </Badge>
              ))}
            </div>
          </div>

          <Badge
            className={`${
              room.gameState.phase === "waiting"
                ? "text-yellow-400 bg-yellow-500/20"
                : room.gameState.phase === "collecting-actions"
                  ? "text-blue-400 bg-blue-500/20"
                  : room.gameState.phase === "processing"
                    ? "text-purple-400 bg-purple-500/20"
                    : "text-green-400 bg-green-500/20"
            }`}
          >
            {room.gameState.phase.replace("-", " ").toUpperCase()}
          </Badge>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {room.gameState.messages.map((message, index) => (
          <div key={index} className="animate-in fade-in duration-300">
            {message.author === "ai" ? (
              <div className="flex gap-3">
                <span className="text-purple-400 font-bold shrink-0">DM:</span>
                <p className="text-gray-100 leading-relaxed">{message.text}</p>
              </div>
            ) : message.author === "system" ? (
              <div className="flex gap-3">
                <span className="text-amber-400 font-bold shrink-0">System:</span>
                <p className="text-gray-300 leading-relaxed italic">{message.text}</p>
              </div>
            ) : (
              <div className="flex gap-3">
                <span className="text-cyan-400 font-bold shrink-0">
                  {room.players.find((p) => p.id === message.playerId)?.name || "Player"}:
                </span>
                <p className="text-gray-300 leading-relaxed">{message.text}</p>
              </div>
            )}
          </div>
        ))}

        {/* Turn Status */}
        {room.gameState.phase === "collecting-actions" && (
          <Card className="border-blue-500/30 bg-blue-500/10">
            <CardContent className="p-4">
              {allPlayersReady ? (
                <div className="text-center">
                  <p className="text-blue-400 font-semibold mb-2">All players ready!</p>
                  <p className="text-gray-300 text-sm">The DM is processing your actions...</p>
                  {isProcessing && (
                    <div className="flex items-center justify-center gap-2 mt-2">
                      <div className="w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
                      <span className="text-xs text-purple-400">Processing turn...</span>
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  <p className="text-blue-400 font-semibold mb-2">
                    Waiting for {waitingForPlayers.length} player(s) to submit their actions:
                  </p>
                  <div className="flex gap-2">
                    {waitingForPlayers.map((player) => (
                      <Badge key={player.id} variant="outline" className="text-xs">
                        {player.name}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {room.gameState.phase === "processing" && (
          <Card className="border-purple-500/30 bg-purple-500/10">
            <CardContent className="p-4 text-center">
              <div className="flex items-center justify-center gap-2 mb-2">
                <div className="w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
                <span className="text-purple-400 font-semibold">DM is weaving your actions into the story...</span>
              </div>
              <p className="text-gray-300 text-sm">Please wait while the AI processes all player actions</p>
            </CardContent>
          </Card>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Action Input */}
      <div className="border-t border-gray-700 bg-[#1A1A1A] p-6">
        <form onSubmit={submitAction} className="flex gap-3">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={
              currentPlayer?.isReady
                ? "Action submitted! Waiting for other players..."
                : room.gameState.phase === "processing"
                  ? "DM is processing actions..."
                  : "What do you do?"
            }
            className="flex-1 bg-gray-800 border-gray-600 text-gray-100 placeholder-gray-400 font-mono focus:border-purple-400 focus:ring-purple-400/20"
            disabled={currentPlayer?.isReady || isSubmitting || room.gameState.phase !== "collecting-actions"}
          />
          <Button
            type="submit"
            disabled={
              !inputValue.trim() ||
              currentPlayer?.isReady ||
              isSubmitting ||
              room.gameState.phase !== "collecting-actions"
            }
            className="bg-purple-600 hover:bg-purple-700 text-white font-mono px-6"
          >
            {isSubmitting ? "..." : currentPlayer?.isReady ? "✓" : "Submit"}
          </Button>
        </form>

        <p className="text-gray-500 text-xs mt-2 font-mono">
          {currentPlayer?.isReady
            ? "Action submitted • Waiting for other players"
            : room.gameState.phase === "processing"
              ? "DM is processing all actions • Please wait"
              : "Submit your action • Turn-based multiplayer"}
        </p>
      </div>
    </div>
  )
}
