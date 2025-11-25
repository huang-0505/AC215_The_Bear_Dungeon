"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import Link from "next/link"

interface Character {
  name: string
  hp: number
  max_hp: number
  ac: number
  attack_bonus: number
  damage: string
  alive: boolean
}

interface CombatState {
  round: number
  current_actor: string | null
  players: Character[]
  enemies: Character[]
  battle_over: boolean
  winner?: string
}

export default function CombatPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [combatSessionId, setCombatSessionId] = useState<string | null>(null)
  const [combatState, setCombatState] = useState<CombatState | null>(null)
  const [narratives, setNarratives] = useState<string[]>([])
  const [actionInput, setActionInput] = useState("")
  const [isPlayerTurn, setIsPlayerTurn] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [showActionPanel, setShowActionPanel] = useState(false)
  const [gameOver, setGameOver] = useState(false)
  const [isInitializing, setIsInitializing] = useState(true)
  const dialogueScrollRef = useRef<HTMLDivElement>(null)
  const enemyTurnInProgress = useRef(false)
  const combatSessionIdRef = useRef<string | null>(null)
  const gameOverRef = useRef(false)

  // Get combat session ID from URL or localStorage
  useEffect(() => {
    // Get session ID from URL params directly (more reliable than searchParams hook)
    const getSessionIdFromUrl = () => {
      if (typeof window !== 'undefined') {
        const params = new URLSearchParams(window.location.search)
        return params.get("session_id")
      }
      return null
    }
    
    const checkSessionId = () => {
      // Try URL first, then localStorage
      const urlSessionId = getSessionIdFromUrl()
      const storedSessionId = localStorage.getItem("combat_session_id")
      const sessionId = urlSessionId || storedSessionId
      
      console.log("Combat page checking for session ID:", {
        urlSessionId,
        storedSessionId,
        finalSessionId: sessionId,
        currentUrl: typeof window !== 'undefined' ? window.location.href : 'N/A'
      })
      
      if (sessionId) {
        console.log("✅ Session ID found:", sessionId)
        setCombatSessionId(sessionId)
        combatSessionIdRef.current = sessionId
        gameOverRef.current = false
        localStorage.setItem("combat_session_id", sessionId)
        setShowActionPanel(true)
        setIsInitializing(false)
        // Load initial state - combat is already started by orchestrator
        // Add a delay to ensure combat session is fully initialized (orchestrator needs time to start it)
        const timer = setTimeout(() => {
          loadCombatState(sessionId, 0)
        }, 1500) // Increased delay to 1.5 seconds
        
        return () => clearTimeout(timer) // Cleanup
      } else {
        // Wait a bit before redirecting - might be a timing issue
        console.log("⚠️ No session ID found yet, waiting 1 second before redirect...")
        const retryTimer = setTimeout(() => {
          const retryUrlId = getSessionIdFromUrl()
          const retryStoredId = localStorage.getItem("combat_session_id")
          const retrySessionId = retryUrlId || retryStoredId
          
          if (retrySessionId) {
            console.log("✅ Session ID found on retry:", retrySessionId)
            setCombatSessionId(retrySessionId)
            combatSessionIdRef.current = retrySessionId
            localStorage.setItem("combat_session_id", retrySessionId)
            setShowActionPanel(true)
            setIsInitializing(false)
            setTimeout(() => {
              loadCombatState(retrySessionId, 0)
            }, 500)
          } else {
            console.error("❌ Still no session ID after retry, redirecting to game")
            setIsInitializing(false)
            window.location.href = "/game" // Use window.location for reliable redirect
          }
        }, 1000) // Wait 1 second before checking again
        
        return () => clearTimeout(retryTimer)
      }
    }
    
    // Initial check
    checkSessionId()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only run once on mount

  const loadCombatState = async (sessionId: string, retryCount = 0) => {
    try {
      console.log("Loading combat state for session:", sessionId, "retry:", retryCount)
      const response = await fetch(`/api/combat/state/${sessionId}`)
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        console.error("Failed to load combat state:", response.status, errorData)
        
        // If 404, the session doesn't exist yet - retry with longer delay
        if (response.status === 404 && retryCount < 5) {
          console.log(`Session not found yet (404), retrying in ${(retryCount + 1) * 1000}ms...`)
          setTimeout(() => {
            loadCombatState(sessionId, retryCount + 1)
          }, (retryCount + 1) * 1000) // Exponential backoff
          return
        }
        
        // Retry up to 5 times for other errors
        if (retryCount < 5) {
          setTimeout(() => {
            loadCombatState(sessionId, retryCount + 1)
          }, 1000)
        } else {
          console.error("Max retries reached, staying on combat page")
          // Don't redirect - show error message instead
          setNarratives(prev => [...prev, `⚠️ Unable to load combat state (${response.status}). The combat session may not be ready yet. Please wait or refresh the page.`])
        }
        return
      }
      
      const data = await response.json()
      console.log("✅ Combat state loaded successfully:", data)
      
      if (!data) {
        console.error("Empty combat state received")
        if (retryCount < 5) {
          setTimeout(() => {
            loadCombatState(sessionId, retryCount + 1)
          }, 1000)
        }
        return
      }
      
      setCombatState(data)
      if (!narratives.length) {
        // Add initial message
        setNarratives(["⚔️ Combat has begun! Roll for initiative!"])
      }
      checkTurn(data)
    } catch (error: any) {
      console.error("Error loading combat state:", error)
      // Retry up to 5 times
      if (retryCount < 5) {
        setTimeout(() => {
          loadCombatState(sessionId, retryCount + 1)
        }, 1000)
      } else {
        console.error("Max retries reached, staying on combat page")
        setNarratives(prev => [...prev, `⚠️ Error loading combat: ${error.message || 'Unknown error'}. Please refresh the page.`])
      }
    }
  }

  const checkTurn = (state: CombatState) => {
    if (!state || !state.current_actor) return

    const currentActor = state.current_actor
    const isPlayer = state.players.some(p => p.name === currentActor && p.alive)
    const isEnemy = state.enemies.some(e => e.name === currentActor && e.alive)

    if (isPlayer) {
      setIsPlayerTurn(true)
      enemyTurnInProgress.current = false // Reset flag when player turn
    } else if (isEnemy) {
      setIsPlayerTurn(false)
      // Only trigger enemy turn if not already in progress and not loading
      // Add a small delay to ensure previous turn processing is complete
      if (!enemyTurnInProgress.current && !isLoading) {
        console.log("[Frontend] Detected enemy turn, will trigger in 1.5s:", currentActor)
        setTimeout(() => {
          // Double-check conditions before triggering
          if (!enemyTurnInProgress.current && !isLoading && combatSessionId && !gameOver) {
            console.log("[Frontend] Timeout fired, calling triggerEnemyTurn()")
            triggerEnemyTurn()
          } else {
            console.log("[Frontend] Conditions changed, skipping enemy turn trigger:", {
              enemyTurnInProgress: enemyTurnInProgress.current,
              isLoading,
              combatSessionId: !!combatSessionId,
              gameOver
            })
          }
        }, 1500)
      } else {
        console.log("[Frontend] Enemy turn already in progress or loading, skipping trigger:", {
          enemyTurnInProgress: enemyTurnInProgress.current,
          isLoading
        })
      }
    }
  }

  const submitAction = async () => {
    if (!actionInput.trim() || !combatSessionId || !isPlayerTurn || isLoading) return

    setIsLoading(true)
    try {
      const response = await fetch(`/api/combat/action/${combatSessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: actionInput.trim() }),
        signal: AbortSignal.timeout(20000), // 20 second timeout
      })

      const data = await response.json()
      
      // Add narrative
      setNarratives(prev => [...prev, data.narrative || data.message || "Action processed"])
      
      // Update state
      if (data.state) {
        setCombatState(data.state)
        
        // Check if battle is over
        if (data.state.battle_over) {
          handleBattleEnd(data.state.winner)
          return
        }
        
        checkTurn(data.state)
      }

      setActionInput("")
    } catch (error: any) {
      console.error("Error submitting action:", error)
      if (error.name === 'AbortError' || error.name === 'TimeoutError') {
        setNarratives(prev => [...prev, "⚠️ Request timed out. Please try again."])
      } else {
        setNarratives(prev => [...prev, `Error: ${error.message || "Failed to submit action"}`])
      }
    } finally {
      setIsLoading(false)
    }
  }

  const triggerEnemyTurn = async () => {
    // Prevent duplicate calls - if already in progress, skip
    if (enemyTurnInProgress.current) {
      console.log("[Frontend] Enemy turn already in progress, skipping duplicate call")
      return
    }

    if (!combatSessionId || gameOver) {
      console.log("[Frontend] Skipping enemy turn - basic conditions not met:", {
        combatSessionId: !!combatSessionId,
        gameOver
      })
      enemyTurnInProgress.current = false
      return
    }

    console.log("[Frontend] Starting enemy turn processing...")
    setIsLoading(true)
    enemyTurnInProgress.current = true
    try {
      console.log("[Frontend] Triggering enemy turn for session:", combatSessionId)
      const response = await fetch(`/api/combat/action/${combatSessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "enemy_turn" }),
        signal: AbortSignal.timeout(20000), // 20 second timeout (frontend timeout should be longer than API timeout)
      })

      console.log("[Frontend] Enemy turn response status:", response.status, response.ok)

      if (!response.ok) {
        const errorText = await response.text()
        console.error("[Frontend] Enemy turn failed:", response.status, errorText)
        let errorData = {}
        try {
          errorData = JSON.parse(errorText)
        } catch {
          errorData = { detail: errorText }
        }
        throw new Error(errorData.detail || `HTTP ${response.status}: ${errorText}`)
      }

      const data = await response.json()
      console.log("[Frontend] Enemy turn response data:", data)
      
      // Add narrative
      setNarratives(prev => [...prev, data.narrative || data.message || "Enemy acts"])
      
      // Reset flag BEFORE checking turn to allow checkTurn to work properly
      enemyTurnInProgress.current = false
      setIsLoading(false)
      
      // Update state
      if (data.state) {
        setCombatState(data.state)
        
        // Check if battle is over
        if (data.state.battle_over) {
          handleBattleEnd(data.state.winner)
          return
        }
        
        // Small delay before checkTurn to ensure state is fully updated
        setTimeout(() => {
          checkTurn(data.state)
        }, 100)
      } else {
        console.warn("[Frontend] Enemy turn response missing state:", data)
        // Try to fetch state manually
        try {
          const stateResponse = await fetch(`/api/combat/state/${combatSessionId}`)
          if (stateResponse.ok) {
            const stateData = await stateResponse.json()
            setCombatState(stateData)
            setTimeout(() => {
              checkTurn(stateData)
            }, 100)
          }
        } catch (e) {
          console.error("[Frontend] Failed to fetch state after enemy turn:", e)
        }
      }
    } catch (error: any) {
      console.error("[Frontend] Error in enemy turn:", error)
      // Reset flag on error too
      enemyTurnInProgress.current = false
      setIsLoading(false)
      
      if (error.name === 'AbortError' || error.name === 'TimeoutError') {
        setNarratives(prev => [...prev, "⚠️ Enemy AI timed out. Trying to recover..."])
        // Try to get current state to continue
        try {
          const stateResponse = await fetch(`/api/combat/state/${combatSessionId}`)
          if (stateResponse.ok) {
            const stateData = await stateResponse.json()
            setCombatState(stateData)
            setTimeout(() => {
              checkTurn(stateData)
            }, 100)
            setNarratives(prev => [...prev, "✅ Recovered combat state. Please continue."])
          } else {
            setNarratives(prev => [...prev, "❌ Failed to recover. Please refresh the page."])
          }
        } catch (e) {
          console.error("[Frontend] Failed to recover state after timeout:", e)
          setNarratives(prev => [...prev, "❌ Failed to recover. Please refresh the page."])
        }
      } else {
        setNarratives(prev => [...prev, `⚠️ Error: ${error.message || "Failed to process enemy turn"}`])
        // Try to recover by fetching state
        try {
          const stateResponse = await fetch(`/api/combat/state/${combatSessionId}`)
          if (stateResponse.ok) {
            const stateData = await stateResponse.json()
            setCombatState(stateData)
            setTimeout(() => {
              checkTurn(stateData)
            }, 100)
          }
        } catch (e) {
          console.error("[Frontend] Failed to recover after error:", e)
        }
      }
    }
  }

  const handleBattleEnd = async (winner?: string) => {
    setIsPlayerTurn(false)
    setActionInput("")
    setGameOver(true)
    gameOverRef.current = true
    
    if (winner === "players") {
      setNarratives(prev => [...prev, "🎉 Victory! 🎉 The heroes have triumphed over their foes!"])
      
      // Note: When we return to game page, the orchestrator will detect combat end
      // on the next action. The combat agent already knows combat ended, so when
      // user returns and takes an action, orchestrator will call combat agent,
      // detect battle_over, and transition to narration automatically.
      
      // Auto-return to game after 3 seconds
      setTimeout(() => {
        localStorage.removeItem("combat_session_id")
        window.location.href = "/game" // Use window.location instead of router
      }, 3000)
    } else {
      setNarratives(prev => [...prev, "💀 Defeat 💀 The enemies have prevailed..."])
      // Game over - players died
      // Stay on page, show game over message
      // The orchestrator will handle game over state when user returns
    }
  }

  const renderCharacter = (char: Character, isActive: boolean) => {
    const hpPercent = (char.hp / char.max_hp) * 100
    return (
      <div
        key={char.name}
        className={`p-3 rounded-lg border-2 transition-all ${
          isActive && char.alive
            ? "border-yellow-500 shadow-lg shadow-yellow-500/50 scale-105"
            : "border-gray-700"
        } ${!char.alive ? "opacity-40 grayscale" : ""} bg-amber-50`}
      >
        <div className="font-bold text-gray-900 mb-1">{char.name}</div>
        <div className="text-sm text-gray-600 mb-2">
          AC: {char.ac} | ATK: +{char.attack_bonus} | DMG: {char.damage}
        </div>
        <div className="w-full h-5 bg-gray-300 rounded-full overflow-hidden border border-gray-700">
          <div
            className="h-full bg-gradient-to-r from-red-800 to-red-500 flex items-center justify-center text-white text-xs font-bold transition-all duration-500"
            style={{ width: `${hpPercent}%` }}
          >
            {char.hp}/{char.max_hp} HP
          </div>
        </div>
      </div>
    )
  }

  // Scroll to bottom when narratives update
  useEffect(() => {
    if (dialogueScrollRef.current) {
      dialogueScrollRef.current.scrollTop = dialogueScrollRef.current.scrollHeight
    }
  }, [narratives])

  // Show loading state while initializing
  if (isInitializing) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-amber-950 via-amber-900 to-amber-950 text-amber-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">⚔️ Loading Combat...</h2>
          <p className="text-amber-200">Preparing the battlefield...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-950 via-amber-900 to-amber-950 text-amber-50 relative overflow-hidden">
      {/* Background overlay */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-0 w-1/2 h-1/2 bg-yellow-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-0 w-1/2 h-1/2 bg-red-500/10 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto p-5 min-h-screen flex flex-col">
        {/* Header */}
        <header className="bg-gradient-to-r from-gray-900 to-gray-700 border-4 border-yellow-600 rounded-lg p-5 mb-5 shadow-lg flex justify-between items-center">
          <h1 className="text-4xl font-bold text-yellow-500 drop-shadow-lg" style={{ fontFamily: "'MedievalSharp', cursive" }}>
            ⚔️ Combat Arena ⚔️
          </h1>
          <div className="flex gap-5 items-center">
            <div className="text-xl font-bold text-amber-50">
              Round: <span className="text-yellow-400">{combatState?.round || 1}</span>
            </div>
            <Link href="/game">
              <Button variant="destructive" className="font-mono">
                Exit Combat
              </Button>
            </Link>
          </div>
        </header>

        {/* Combat Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 flex-1 mb-5">
          {/* Left: Players */}
          <div className="bg-amber-50 border-4 border-gray-900 rounded-lg p-4 shadow-lg overflow-y-auto max-h-[600px]">
            <h2 className="text-2xl font-bold text-center mb-4 pb-2 border-b-2 border-gray-900 text-gray-900" style={{ fontFamily: "'MedievalSharp', cursive" }}>
              ⚔️ Heroes
            </h2>
            <div className="space-y-3">
              {combatState?.players.map((player) =>
                renderCharacter(player, player.name === combatState.current_actor)
              )}
            </div>
          </div>

          {/* Center: Dialogue */}
          <div className="bg-amber-50 border-4 border-gray-900 rounded-lg p-4 shadow-lg overflow-y-auto max-h-[600px]">
            <div
              ref={dialogueScrollRef}
              className="space-y-3 min-h-full"
            >
              {narratives.length === 0 ? (
                <div className="text-center py-10">
                  <h2 className="text-2xl font-bold text-gray-900 mb-3" style={{ fontFamily: "'MedievalSharp', cursive" }}>Welcome to Combat!</h2>
                  <p className="text-gray-600 mb-5">Loading combat state...</p>
                </div>
              ) : (
                narratives.map((narrative, idx) => (
                  <div
                    key={idx}
                    className="bg-white/80 border-l-4 border-yellow-600 p-4 rounded shadow-sm animate-in fade-in"
                  >
                    <p className="text-gray-900 leading-relaxed">{narrative}</p>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Right: Enemies */}
          <div className="bg-amber-50 border-4 border-gray-900 rounded-lg p-4 shadow-lg overflow-y-auto max-h-[600px]">
            <h2 className="text-2xl font-bold text-center mb-4 pb-2 border-b-2 border-gray-900 text-gray-900" style={{ fontFamily: "'MedievalSharp', cursive" }}>
              💀 Foes
            </h2>
            <div className="space-y-3">
              {combatState?.enemies.map((enemy) =>
                renderCharacter(enemy, enemy.name === combatState.current_actor)
              )}
            </div>
          </div>
        </div>

        {/* Action Panel */}
        {showActionPanel && combatState && !combatState.battle_over && !gameOver && (
          <div className="bg-amber-50 border-4 border-gray-900 rounded-lg p-5 shadow-lg">
            <div className={`text-center text-xl font-bold mb-4 p-3 rounded ${
              isPlayerTurn ? "bg-green-100 text-green-900" : "bg-red-100 text-red-900"
            }`}>
              {isPlayerTurn
                ? `${combatState.current_actor}'s Turn - Choose your action!`
                : `${combatState.current_actor}'s Turn - Enemy is deciding...`}
            </div>
            <div className="flex gap-3">
              <Input
                value={actionInput}
                onChange={(e) => setActionInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === "Enter" && !isLoading && isPlayerTurn) {
                    submitAction()
                  }
                }}
                placeholder="Describe your action... (e.g., 'I swing my sword at the goblin')"
                disabled={!isPlayerTurn || isLoading}
                className="flex-1 bg-amber-100 border-2 border-gray-700 text-gray-900 placeholder-gray-500 focus:border-yellow-600"
                style={{ fontFamily: "'Cinzel', serif" }}
              />
              <Button
                onClick={submitAction}
                disabled={!actionInput.trim() || !isPlayerTurn || isLoading}
                className="bg-green-700 hover:bg-green-800 text-white font-mono px-6"
              >
                {isLoading ? "..." : "⚔️ Take Action"}
              </Button>
            </div>
          </div>
        )}

        {/* Game Over Message */}
        {gameOver && combatState?.battle_over && (
          <div className="bg-gradient-to-r from-yellow-600 to-amber-500 border-4 border-gray-900 rounded-lg p-8 text-center shadow-lg">
            <h2 className="text-4xl font-bold text-gray-900 mb-4" style={{ fontFamily: "'MedievalSharp', cursive" }}>
              {combatState.winner === "players" ? "🎉 Victory! 🎉" : "💀 Defeat 💀"}
            </h2>
            <p className="text-xl text-gray-800 mb-4">
              {combatState.winner === "players" 
                ? "The heroes have triumphed over their foes!" 
                : "The enemies have prevailed..."}
            </p>
            {combatState.winner === "players" && (
              <p className="text-gray-700">Returning to game...</p>
            )}
          </div>
        )}
      </div>

      {/* Add Google Fonts */}
      <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=MedievalSharp&display=swap" rel="stylesheet" />
    </div>
  )
}

