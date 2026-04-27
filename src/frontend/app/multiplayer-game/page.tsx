"use client"

import type React from "react"

import { useEffect, useMemo, useRef, useState } from "react"
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { ArrowLeft, CheckCircle, Circle, Clock, Crown, Users } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useRoomEvents } from "@/hooks/use-room-events"

interface NarrativeEntry {
  round: number
  text: string
}

interface CampaignSummary {
  id: string
  name: string
}

export default function MultiplayerGame() {
  const router = useRouter()
  const params = useSearchParams()
  const roomIdParam = params.get("roomId")
  const playerIdParam = params.get("playerId")

  const [roomId, setRoomId] = useState<string | null>(roomIdParam)
  const [playerId, setPlayerId] = useState<string | null>(playerIdParam)

  const {
    room,
    isConnected,
    submittedPlayerIds,
    lastResult,
    isEnded,
    actingPlayerId,
    stateType,
  } = useRoomEvents(roomId)

  const [inputValue, setInputValue] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [narrativeLog, setNarrativeLog] = useState<NarrativeEntry[]>([])
  const [campaigns, setCampaigns] = useState<CampaignSummary[]>([])
  const [selectedCampaign, setSelectedCampaign] = useState<string>("")
  const [starting, setStarting] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Hydrate identity from localStorage if not in URL. If roomId, playerId,
  // OR the per-room token can't be resolved, send the user back to the
  // lobby — otherwise the page would render with null identifiers and
  // silently mis-attribute every action (or 401 on every submit).
  useEffect(() => {
    let nextRoom = roomId
    if (!nextRoom) {
      nextRoom = localStorage.getItem("roomId")
      if (nextRoom) setRoomId(nextRoom)
    } else {
      localStorage.setItem("roomId", nextRoom)
    }

    let nextPlayer = playerId
    if (!nextPlayer) {
      nextPlayer = localStorage.getItem("playerId")
      if (nextPlayer) setPlayerId(nextPlayer)
    }

    const hasToken =
      nextRoom !== null && sessionStorage.getItem(`playerToken:${nextRoom}`) !== null

    if (!nextRoom || !nextPlayer || !hasToken) {
      router.push("/multiplayer")
    }
  }, [roomId, playerId, router])

  // Load campaign list once for the host's start dialog.
  useEffect(() => {
    let cancelled = false
    fetch("/api/orchestrator/campaigns")
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (cancelled || !data) return
        const list: CampaignSummary[] = (data.campaigns ?? []).map(
          (c: { campaign_id?: string; id?: string; name: string }) => ({
            id: c.campaign_id ?? c.id ?? "",
            name: c.name,
          }),
        )
        setCampaigns(list.filter((c) => c.id))
      })
      .catch(() => {
        /* campaign list is optional */
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Append new narration whenever a round resolves.
  useEffect(() => {
    if (!lastResult?.response) return
    setNarrativeLog((prev) => [
      ...prev,
      { round: lastResult.current_round, text: lastResult.response },
    ])
    setInputValue("")
  }, [lastResult])

  // Auto-scroll the narrative log.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [narrativeLog])

  const isHost = useMemo(
    () => room && playerId && room.host_player_id === playerId,
    [room, playerId],
  )

  const hasSubmitted = useMemo(
    () => (playerId ? submittedPlayerIds.includes(playerId) : false),
    [playerId, submittedPlayerIds],
  )

  const inCombat = stateType === "combat"
  const isMyCombatTurn = inCombat && actingPlayerId === playerId
  const actingPlayer = useMemo(
    () =>
      room?.players.find((p) => p.player_id === actingPlayerId) ?? null,
    [room, actingPlayerId],
  )

  const inputDisabled = isEnded
    ? true
    : inCombat
      ? !isMyCombatTurn || submitting
      : hasSubmitted || submitting

  const placeholderText = isEnded
    ? "The adventure has ended."
    : inCombat
      ? isMyCombatTurn
        ? "It's your turn — what do you do?"
        : actingPlayer
          ? `Waiting for ${actingPlayer.character_name}'s turn…`
          : "Waiting for the next combatant…"
      : hasSubmitted
        ? "Action submitted! Waiting for other players…"
        : "What do you do?"

  const startAdventure = async () => {
    if (!roomId || !playerId) return
    const token = sessionStorage.getItem(`playerToken:${roomId}`)
    if (!token) {
      alert("Missing player token; please rejoin from the lobby")
      router.push("/multiplayer")
      return
    }
    setStarting(true)
    try {
      const response = await fetch(`/api/rooms/${roomId}/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Player-Token": token,
        },
        body: JSON.stringify({
          player_id: playerId,
          campaign_id: selectedCampaign || undefined,
        }),
      })
      if (!response.ok) {
        const error = await response.json()
        alert(error.detail || "Failed to start adventure")
        return
      }
      const data = await response.json()
      if (data.response) {
        setNarrativeLog((prev) => [
          ...prev,
          { round: 0, text: data.response },
        ])
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error("Error starting adventure", error)
    } finally {
      setStarting(false)
    }
  }

  const submitAction = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!inputValue.trim() || !roomId || !playerId || hasSubmitted) return
    const token = sessionStorage.getItem(`playerToken:${roomId}`)
    if (!token) {
      alert("Missing player token; please rejoin from the lobby")
      router.push("/multiplayer")
      return
    }
    setSubmitting(true)
    try {
      const response = await fetch(`/api/rooms/${roomId}/action`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Player-Token": token,
        },
        body: JSON.stringify({ player_id: playerId, text: inputValue.trim() }),
      })
      if (!response.ok) {
        const error = await response.json()
        alert(error.detail || "Action rejected")
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error("Error submitting action", error)
    } finally {
      setSubmitting(false)
    }
  }

  if (!roomId) {
    return null
  }

  if (!room) {
    return (
      <div className="h-screen bg-[#1A1A1A] text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-purple-400 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">
            {isConnected ? "Loading room…" : "Connecting to room…"}
          </p>
        </div>
      </div>
    )
  }

  const partySize = room.players.length
  const submittedCount = submittedPlayerIds.length
  const waitingPlayers = room.players.filter(
    (p) => !submittedPlayerIds.includes(p.player_id),
  )

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
                Round {room.current_round}
              </Badge>
              <Badge
                className={
                  room.state === "lobby"
                    ? "text-yellow-400 bg-yellow-500/20"
                    : room.state === "active"
                      ? "text-blue-400 bg-blue-500/20"
                      : "text-gray-400 bg-gray-500/20"
                }
              >
                {room.state.toUpperCase()}
              </Badge>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-400" : "bg-red-400"}`}
            />
            <span className="text-xs text-gray-400">
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>
      </div>

      {/* Players panel */}
      <div className="border-b border-gray-700 bg-gray-800/50 p-4">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-sm text-gray-400">Party:</span>
            {room.players.map((p) => {
              const submitted = submittedPlayerIds.includes(p.player_id)
              const isMe = p.player_id === playerId
              const host = room.host_player_id === p.player_id
              return (
                <Badge
                  key={p.player_id}
                  variant={isMe ? "default" : "outline"}
                  className={`text-xs flex items-center gap-1 ${
                    submitted
                      ? "border-green-400 text-green-400"
                      : "border-gray-500 text-gray-400"
                  }`}
                >
                  {submitted ? (
                    <CheckCircle className="w-3 h-3" />
                  ) : (
                    <Circle className="w-3 h-3" />
                  )}
                  {host && <Crown className="w-3 h-3 text-yellow-400" />}
                  {p.character_name} ({p.character_class})
                </Badge>
              )
            })}
          </div>

          {room.state === "active" && (
            <span className="text-xs text-gray-400">
              {submittedCount}/{partySize} submitted
            </span>
          )}
        </div>
      </div>

      {/* Lobby start UI (host only) */}
      {room.state === "lobby" && (
        <div className="border-b border-gray-700 bg-yellow-500/10 p-4">
          {isHost ? (
            <div className="flex items-center gap-3 flex-wrap">
              <span className="text-sm text-yellow-400">
                You are the host. Pick a campaign and start when ready.
              </span>
              <Select value={selectedCampaign} onValueChange={setSelectedCampaign}>
                <SelectTrigger className="w-64 bg-gray-800 border-gray-600">
                  <SelectValue placeholder="Default sandbox" />
                </SelectTrigger>
                <SelectContent>
                  {campaigns.map((c) => (
                    <SelectItem key={c.id} value={c.id}>
                      {c.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button onClick={startAdventure} disabled={starting}>
                {starting ? "Starting…" : "Start Adventure"}
              </Button>
            </div>
          ) : (
            <p className="text-sm text-yellow-400">
              Waiting for the host to start the adventure…
            </p>
          )}
        </div>
      )}

      {/* Narrative log */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {narrativeLog.length === 0 && room.state !== "lobby" && (
          <p className="text-gray-500 text-sm">
            The DM is preparing your scene…
          </p>
        )}

        {narrativeLog.map((entry, index) => (
          <div key={index} className="animate-in fade-in duration-300">
            <div className="flex gap-3">
              <span className="text-purple-400 font-bold shrink-0">
                DM • R{entry.round}
              </span>
              <p className="text-gray-100 leading-relaxed whitespace-pre-wrap">
                {entry.text}
              </p>
            </div>
          </div>
        ))}

        {room.state === "active" && (
          <Card className="border-blue-500/30 bg-blue-500/10">
            <CardContent className="p-4">
              {submittedCount === partySize ? (
                <div className="text-center">
                  <p className="text-blue-400 font-semibold mb-2">
                    All players ready!
                  </p>
                  <p className="text-gray-300 text-sm">
                    The DM is weaving your actions into the story…
                  </p>
                </div>
              ) : waitingPlayers.length > 0 ? (
                <div>
                  <p className="text-blue-400 font-semibold mb-2">
                    Waiting for {waitingPlayers.length} player(s):
                  </p>
                  <div className="flex gap-2 flex-wrap">
                    {waitingPlayers.map((p) => (
                      <Badge
                        key={p.player_id}
                        variant="outline"
                        className="text-xs"
                      >
                        {p.character_name}
                      </Badge>
                    ))}
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        )}

        {isEnded && (
          <Card className="border-green-500/30 bg-green-500/10">
            <CardContent className="p-4 text-center">
              <p className="text-green-400 font-semibold">
                The adventure has ended.
              </p>
            </CardContent>
          </Card>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Action input */}
      {room.state === "active" && !isEnded && (
        <div className="border-t border-gray-700 bg-[#1A1A1A] p-6">
          {inCombat && (
            <div className="mb-3 flex items-center gap-2">
              <Badge
                className={
                  isMyCombatTurn
                    ? "bg-purple-600 text-white"
                    : "bg-gray-700 text-gray-300"
                }
              >
                {isMyCombatTurn
                  ? "Your turn"
                  : actingPlayer
                    ? `${actingPlayer.character_name}'s turn`
                    : "Combat in progress"}
              </Badge>
            </div>
          )}

          <form onSubmit={submitAction} className="flex gap-3">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={placeholderText}
              className="flex-1 bg-gray-800 border-gray-600 text-gray-100 placeholder-gray-400 font-mono focus:border-purple-400 focus:ring-purple-400/20"
              disabled={inputDisabled}
            />
            <Button
              type="submit"
              disabled={!inputValue.trim() || inputDisabled}
              className="bg-purple-600 hover:bg-purple-700 text-white font-mono px-6"
            >
              {submitting
                ? "…"
                : inCombat
                  ? isMyCombatTurn
                    ? "Act"
                    : "Wait"
                  : hasSubmitted
                    ? "✓"
                    : "Submit"}
            </Button>
          </form>

          <p className="text-gray-500 text-xs mt-2 font-mono">
            {inCombat
              ? isMyCombatTurn
                ? "Your turn • Resolve and pass to the next combatant"
                : "Strict turn order • Wait for the acting combatant"
              : hasSubmitted
                ? "Action submitted • Waiting for the rest of the party"
                : "Submit your action • All players resolve together"}
          </p>
        </div>
      )}
    </div>
  )
}
