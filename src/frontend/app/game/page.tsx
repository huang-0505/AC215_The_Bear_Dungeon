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
  choices?: string[]  // Tree-structure mode: suggested choices
  combat_available?: boolean  // Whether combat is available
  isEnding?: boolean  // Whether this is an ending message
  endingType?: string  // Type of ending (victory, defeat, neutral)
  narrationRound?: number  // Current narration round
  combatCount?: number  // Number of combats completed
  maxCombats?: number  // Maximum combats before game ends
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

// Helper function to safely access localStorage
const safeLocalStorage = {
  getItem: (key: string): string | null => {
    if (typeof window === 'undefined') return null
    try {
      return localStorage.getItem(key)
    } catch (e) {
      console.error('localStorage.getItem error:', e)
      return null
    }
  },
  setItem: (key: string, value: string): void => {
    if (typeof window === 'undefined') return
    try {
      localStorage.setItem(key, value)
    } catch (e) {
      console.error('localStorage.setItem error:', e)
    }
  },
  removeItem: (key: string): void => {
    if (typeof window === 'undefined') return
    try {
      localStorage.removeItem(key)
    } catch (e) {
      console.error('localStorage.removeItem error:', e)
    }
  }
}

export default function GameInterface() {
  const [characterClass, setCharacterClass] = useState<string>("")
  const [selectedCampaign, setSelectedCampaign] = useState<string>("")
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState("")
  const [isAiThinking, setIsAiThinking] = useState(false)
  const [showUpload, setShowUpload] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [currentRound, setCurrentRound] = useState<number | undefined>(undefined)
  const [currentCombatCount, setCurrentCombatCount] = useState<number | undefined>(undefined)
  const [currentMaxCombats, setCurrentMaxCombats] = useState<number | undefined>(undefined)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Ensure we're in the browser
    if (typeof window === 'undefined') return

    const selectedClass = safeLocalStorage.getItem("selectedCharacterClass") || "Fighter"
    const campaign = safeLocalStorage.getItem("selectedCampaign") || "classic-dungeon"
    setCharacterClass(selectedClass)
    setSelectedCampaign(campaign)

    // Check if we're returning from combat (not starting a fresh game)
    const returningFromCombat = safeLocalStorage.getItem("returning_from_combat") === "true"
    const savedSessionId = safeLocalStorage.getItem("game_session_id")
    const savedMessages = safeLocalStorage.getItem("game_messages")
    
    // Only restore session if we're returning from combat
    // Otherwise, clear old session data and start fresh
    if (!returningFromCombat) {
      console.log("Starting fresh game - clearing old session data")
      safeLocalStorage.removeItem("game_session_id")
      safeLocalStorage.removeItem("game_messages")
      safeLocalStorage.removeItem("combat_session_id")
    }
    
    // Async function to restore session (only called if returning from combat)
    async function restoreSession() {
      if (returningFromCombat && savedSessionId && savedMessages) {
        console.log("Returning from combat, attempting to restore session...")
        setSessionId(savedSessionId)
        try {
          const parsedMessages = JSON.parse(savedMessages)
          setMessages(parsedMessages)
          
          // Verify session is still valid by checking game state
          // Add timeout and better error handling
          let stateResponse: Response | null = null
          try {
            stateResponse = await fetch(`/api/orchestrator/game/state/${savedSessionId}`, {
              signal: AbortSignal.timeout(5000) // 5 second timeout
            })
          } catch (fetchError: any) {
            console.warn("Session validation failed (network/timeout), but continuing with restoration:", fetchError)
            // Don't fail completely - try to restore anyway
            // The session might still be valid, just the API call failed
            stateResponse = null
          }
          
          if (stateResponse && stateResponse.ok) {
            const stateData = await stateResponse.json()
            const currentState = stateData.current_state
            
            // ✅ CRITICAL: Get round/combat count from tree data, not just node metadata
            const treeData = stateData.full_tree || {}
            const narrationRound = typeof treeData.narration_round === 'number' ? treeData.narration_round : 
                                  (currentState?.metadata?.narration_round || 0)
            const combatCount = typeof treeData.combat_count === 'number' ? treeData.combat_count : 
                               (currentState?.metadata?.combat_count || 0)
            const maxCombats = typeof treeData.max_combats === 'number' ? treeData.max_combats : 
                              (currentState?.metadata?.max_combats || 5)
            
            console.log("Session restored successfully, current state:", currentState?.state_type, 
                       `Round: ${narrationRound}, Combats: ${combatCount}/${maxCombats}`)
            
            // ✅ Update all existing messages with correct round/combat count info
            // This ensures the UI shows the correct counts
            const updatedMessagesWithCounts = parsedMessages.map((msg: Message) => ({
              ...msg,
              narrationRound: msg.narrationRound !== undefined ? msg.narrationRound : narrationRound,
              combatCount: msg.combatCount !== undefined ? msg.combatCount : combatCount,
              maxCombats: msg.maxCombats !== undefined ? msg.maxCombats : maxCombats,
            }))
            setMessages(updatedMessagesWithCounts)
            safeLocalStorage.setItem("game_messages", JSON.stringify(updatedMessagesWithCounts))
            
            // ✅ Update current round/combat state for UI display
            setCurrentRound(narrationRound)
            setCurrentCombatCount(combatCount)
            setCurrentMaxCombats(maxCombats)
            
            // Check if we're in narration state after combat (combat just ended)
            // This works for EVERY combat, not just the first one
            if (currentState && currentState.state_type === "narration") {
              // Check if this is a post-combat narration node
              const isPostCombat = currentState.metadata?.combat_outcome || 
                                   currentState.metadata?.previous_combat_id ||
                                   // Also check if the last message was about combat
                                   (parsedMessages.length > 0 && 
                                    parsedMessages[parsedMessages.length - 1]?.text?.includes("⚔️"))
              
              if (isPostCombat) {
                // This is a post-combat narration node - combat just ended
                const combatOutcome = currentState.metadata?.combat_outcome
                const agentResponse = currentState.agent_response || 
                  currentState.narrative_text ||
                  (combatOutcome === "players" 
                    ? "🎉 Victory! The battle has been won. The adventure continues..."
                    : "The battle has ended. The adventure continues...")
                
                // Extract choices from metadata or state
                const choices = currentState.metadata?.choices || 
                               currentState.metadata?.story_choices || 
                               []
                
                const postCombatMessage: Message = {
                  author: "ai",
                  text: agentResponse,
                  timestamp: Date.now(),
                  choices: Array.isArray(choices) && choices.length > 0 ? choices : undefined,
                  combat_available: currentState.metadata?.combat_available !== false,
                  narrationRound: narrationRound,
                  combatCount: combatCount,
                  maxCombats: maxCombats,
                }
                
                // Check if this message is already in the messages list
                // Compare by text content to avoid duplicates
                const lastMessage = updatedMessagesWithCounts[updatedMessagesWithCounts.length - 1]
                const messageExists = lastMessage && 
                  lastMessage.author === "ai" && 
                  (lastMessage.text === agentResponse || 
                   lastMessage.text.includes(agentResponse.substring(0, 50)) ||
                   (agentResponse.length > 50 && lastMessage.text.includes(agentResponse.substring(0, 50))))
                
                if (!messageExists) {
                  // Add the post-combat narration to messages
                  const finalMessages = [...updatedMessagesWithCounts, postCombatMessage]
                  setMessages(finalMessages)
                  safeLocalStorage.setItem("game_messages", JSON.stringify(finalMessages))
                  console.log("✅ Added post-combat narration to messages (combat #" + combatCount + "):", agentResponse.substring(0, 50))
                } else {
                  console.log("Post-combat narration already in messages, skipping duplicate")
                }
              } else {
                // Regular narration state - messages already restored with correct counts
                console.log("In narration state, messages restored with round/combat counts")
              }
            } else if (currentState && currentState.state_type === "combat") {
              // Still in combat state - this shouldn't happen if orchestrator was notified
              // But handle it gracefully
              console.warn("Still in combat state after returning from combat - orchestrator may not have processed yet")
              // Keep messages as-is, user can continue
            }
            
            // Clear the returning flag and combat session ID (works for every combat)
            safeLocalStorage.removeItem("returning_from_combat")
            safeLocalStorage.removeItem("combat_session_id")
            console.log("✅ Session restored successfully after combat - Round:", narrationRound, "Combats:", combatCount)
            return true // Session restored successfully
          } else if (stateResponse === null) {
            // Network error but we'll try to continue anyway
            console.log("Session validation failed (network error), but restoring messages anyway")
            // Messages already restored above, just clear flags
            safeLocalStorage.removeItem("returning_from_combat")
            safeLocalStorage.removeItem("combat_session_id")
            return true // Restore anyway - user can continue
          } else {
            // State check failed (404 or other error)
            // But we should still try to restore messages if they exist
            // The session might still be valid, just the API call failed
            console.warn("Session state check failed, but attempting to restore messages anyway")
            if (parsedMessages && parsedMessages.length > 0) {
              // Restore messages - user can try to continue
              setMessages(parsedMessages)
              safeLocalStorage.setItem("game_messages", JSON.stringify(parsedMessages))
              safeLocalStorage.removeItem("returning_from_combat")
              safeLocalStorage.removeItem("combat_session_id")
              console.log("✅ Restored messages despite state check failure - user can try to continue")
              return true // Partial restoration - better than nothing
            } else {
              // No messages to restore, session is truly invalid
              console.log("Session no longer valid and no messages to restore, starting new game")
              safeLocalStorage.removeItem("game_session_id")
              safeLocalStorage.removeItem("game_messages")
              safeLocalStorage.removeItem("returning_from_combat")
              return false
            }
          }
        } catch (e) {
          console.error("Error restoring session:", e)
          // Try to preserve messages even if there's an error
          // User might still be able to continue
          try {
            const parsedMessages = JSON.parse(savedMessages)
            setMessages(parsedMessages)
            console.log("Restored messages despite error, user can try to continue")
            safeLocalStorage.removeItem("returning_from_combat")
            safeLocalStorage.removeItem("combat_session_id")
            return true // Partial restoration
          } catch (parseError) {
            console.error("Failed to parse messages:", parseError)
          }
          // Clear invalid data and fall through to start new game
          safeLocalStorage.removeItem("game_session_id")
          safeLocalStorage.removeItem("game_messages")
          safeLocalStorage.removeItem("returning_from_combat")
          return false
        }
      }
      return false
    }
    
    // Try to restore session first, then start new game if needed
    restoreSession().then((restored) => {
      if (restored) {
        return // Session restored, exit early
      }
      
      // Clear combat session ID and returning flag if they exist (cleanup)
      safeLocalStorage.removeItem("combat_session_id")
      safeLocalStorage.removeItem("returning_from_combat")

      // Start new game session with orchestrator
      async function startGameSession() {
        try {
          const response = await fetch('/api/game/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              campaign_id: campaign,
              character_class: selectedClass,
              character_name: selectedClass, // Use class name as default
              max_combats: 5,           // Game ends after 5 combats (default: 5)
              combat_rounds: [3, 5, 10, 15]  // Combat available at these rounds (default: [3, 5, 10, 15])
            }),
          }).catch((fetchError) => {
            console.error('Fetch error:', fetchError)
            throw new Error(`Network error: ${fetchError.message}`)
          })

          if (!response.ok) {
            const errorText = await response.text().catch(() => 'Unknown error')
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`)
          }
            
          const data = await response.json().catch((parseError) => {
            console.error('JSON parse error:', parseError)
            throw new Error('Invalid JSON response from server')
          })
          
          if (!data || !data.session_id) {
            throw new Error("Invalid response from server: missing session_id")
          }
          
          setSessionId(data.session_id)
          safeLocalStorage.setItem("game_session_id", data.session_id)

          // Add initial narrative as first message
          const narrationRound = typeof data.narration_round === 'number' ? data.narration_round : 0
          const combatCount = typeof data.combat_count === 'number' ? data.combat_count : 0
          const maxCombats = typeof data.max_combats === 'number' ? data.max_combats : 5
          
          // ✅ Set current state
          setCurrentRound(narrationRound)
          setCurrentCombatCount(combatCount)
          setCurrentMaxCombats(maxCombats)
          
          const initialMessage: Message = {
            author: "ai",
            text: data.response || data.message || "Welcome to the adventure!",
            timestamp: Date.now(),
            choices: Array.isArray(data.choices) ? data.choices : undefined,
            combat_available: data.combat_available === true,
            narrationRound: narrationRound,
            combatCount: combatCount,
            maxCombats: maxCombats,
          }
          setMessages([initialMessage])
          safeLocalStorage.setItem("game_messages", JSON.stringify([initialMessage]))
        } catch (error: any) {
          console.error('Failed to start game session:', error)
          // Fallback to local opening
          try {
            const characterOpening = getCharacterOpening(selectedClass, campaign)
            const errorMessage = error?.message || 'Unknown error'
            const initialMessage: Message = {
              author: "ai",
              text: `${characterOpening}\n\n⚠️ Failed to connect to game server - using offline mode.\nError: ${errorMessage}`,
              timestamp: Date.now(),
            }
            setMessages([initialMessage])
            safeLocalStorage.setItem("game_messages", JSON.stringify([initialMessage]))
          } catch (fallbackError) {
            console.error('Error in fallback:', fallbackError)
            // Last resort - just show a simple message
            setMessages([{
              author: "ai",
              text: "⚠️ An error occurred while starting the game. Please refresh the page.",
              timestamp: Date.now(),
            }])
          }
        }
      }

      startGameSession()
    })
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleChoiceClick = async (choice: string) => {
    if (isAiThinking || !sessionId) return
    
    // Add player message
    const playerMessage: Message = {
      author: "player",
      text: choice,
      timestamp: Date.now(),
    }

    setMessages((prev) => {
      const updated = [...prev, playerMessage]
      // Save messages to localStorage
      safeLocalStorage.setItem("game_messages", JSON.stringify(updated))
      return updated
    })
    setIsAiThinking(true)

    try {
      const response = await fetch("/api/game/action", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          text: choice,
        }),
      })

      if (!response.ok) {
        // If 404, the session doesn't exist - clear it and show error
        if (response.status === 404) {
          console.error('Session not found (404), clearing localStorage')
          safeLocalStorage.removeItem("game_session_id")
          safeLocalStorage.removeItem("game_messages")
          safeLocalStorage.removeItem("combat_session_id")
          setSessionId(null)
          setIsAiThinking(false)
          const errorMessage: Message = {
            author: "ai",
            text: "⚠️ Your game session has expired. Please refresh the page to start a new game.",
            timestamp: Date.now(),
          }
          setMessages((prev) => {
            const updated = [...prev, errorMessage]
            safeLocalStorage.setItem("game_messages", JSON.stringify(updated))
            return updated
          })
          return
        }
        const errorText = await response.text().catch(() => 'Unknown error')
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`)
      }

      const data = await response.json()
      
      if (!data) {
        throw new Error("Empty response from server")
      }

      // Handle state transition to combat FIRST - redirect immediately
      if (data && data.state_type === "combat" && data.combat_session_id) {
        safeLocalStorage.setItem("combat_session_id", data.combat_session_id)
        setIsAiThinking(false)
        // Redirect to dedicated combat UI immediately (don't add message to state)
        // Use replace to prevent back button issues
        window.location.replace(`/game/combat?session_id=${data.combat_session_id}`)
        return
      }

      if (!data.response) {
        throw new Error("Invalid response from server: missing response field")
      }

      let responseText = data.response || ""

      // Add rule validation feedback if action was invalid
      if (data.validation && !data.validation.is_valid) {
        responseText = `⚠️  **Rule Check:** ${data.validation.explanation || "Invalid action"}\n\n${responseText}`
      }

      const aiMessage: Message = {
        author: "ai",
        text: responseText,
        timestamp: Date.now(),
        choices: Array.isArray(data.choices) ? data.choices : undefined,
        combat_available: data.combat_available === true,
        isEnding: data.is_ending === true,
        endingType: typeof data.ending_type === 'string' ? data.ending_type : undefined,
        narrationRound: typeof data.narration_round === 'number' ? data.narration_round : undefined,
        combatCount: typeof data.combat_count === 'number' ? data.combat_count : undefined,
        maxCombats: typeof data.max_combats === 'number' ? data.max_combats : undefined,
      }

      setMessages((prev) => {
        const updated = [...prev, aiMessage]
        // Save messages to localStorage
        safeLocalStorage.setItem("game_messages", JSON.stringify(updated))
        return updated
      })
      
      // Reset thinking state after successful response
      setIsAiThinking(false)
      
      // Handle game ending
      if (data.is_ending) {
        setInputValue("")
        console.log("Game ended:", data.ending_type)
      }
    } catch (error: any) {
      console.error("Error calling orchestrator:", error)
      setIsAiThinking(false)
      const errorMessage: Message = {
        author: "ai",
        text: `⚠️ Error: ${error?.message || "Failed to process action. Please try again."}`,
        timestamp: Date.now(),
      }
      setMessages((prev) => {
        const updated = [...prev, errorMessage]
        safeLocalStorage.setItem("game_messages", JSON.stringify(updated))
        return updated
      })
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || isAiThinking || !sessionId) return

    // Add player message
    const playerMessage: Message = {
      author: "player",
      text: inputValue.trim(),
      timestamp: Date.now(),
    }

    setMessages((prev) => {
      const updated = [...prev, playerMessage]
      // Save messages to localStorage
      safeLocalStorage.setItem("game_messages", JSON.stringify(updated))
      return updated
    })
    setInputValue("")
    setIsAiThinking(true)

    try {
      if (!sessionId) {
        console.error("No session ID available")
        setIsAiThinking(false)
        return
      }

      const response = await fetch("/api/game/action", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          text: inputValue.trim(),
        }),
      })

      if (!response.ok) {
        // If 404, the session doesn't exist - clear it and show error
        if (response.status === 404) {
          console.error('Session not found (404), clearing localStorage')
          safeLocalStorage.removeItem("game_session_id")
          safeLocalStorage.removeItem("game_messages")
          safeLocalStorage.removeItem("combat_session_id")
          setSessionId(null)
          setIsAiThinking(false)
          setInputValue("")
          const errorMessage: Message = {
            author: "ai",
            text: "⚠️ Your game session has expired. Please refresh the page to start a new game.",
            timestamp: Date.now(),
          }
          setMessages((prev) => {
            const updated = [...prev, errorMessage]
            safeLocalStorage.setItem("game_messages", JSON.stringify(updated))
            return updated
          })
          return
        }
        const errorText = await response.text().catch(() => 'Unknown error')
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`)
      }

      const data = await response.json()
      
      if (!data) {
        throw new Error("Empty response from server")
      }

      // Handle state transition to combat FIRST - redirect immediately
      if (data && data.state_type === "combat" && data.combat_session_id) {
        safeLocalStorage.setItem("combat_session_id", data.combat_session_id)
        setIsAiThinking(false)
        // Redirect to dedicated combat UI immediately (don't add message to state)
        // Use replace to prevent back button issues
        window.location.replace(`/game/combat?session_id=${data.combat_session_id}`)
        return
      }

      if (!data || !data.response) {
        throw new Error("Invalid response from server")
      }

      let responseText = data.response || ""

      // Add rule validation feedback if action was invalid
      if (data.validation && !data.validation.is_valid) {
        responseText = `⚠️  **Rule Check:** ${data.validation.explanation || "Invalid action"}\n\n${responseText}`
      }

      // ✅ Get round/combat counts from response, fallback to current state
      const narrationRound = typeof data.narration_round === 'number' ? data.narration_round : currentRound
      const combatCount = typeof data.combat_count === 'number' ? data.combat_count : currentCombatCount
      const maxCombats = typeof data.max_combats === 'number' ? data.max_combats : currentMaxCombats
      
      // ✅ Update current state
      if (narrationRound !== undefined) setCurrentRound(narrationRound)
      if (combatCount !== undefined) setCurrentCombatCount(combatCount)
      if (maxCombats !== undefined) setCurrentMaxCombats(maxCombats)

      const aiMessage: Message = {
        author: "ai",
        text: responseText,
        timestamp: Date.now(),
        choices: Array.isArray(data.choices) ? data.choices : undefined,
        combat_available: data.combat_available === true,
        isEnding: data.is_ending === true,
        endingType: typeof data.ending_type === 'string' ? data.ending_type : undefined,
        narrationRound: narrationRound,
        combatCount: combatCount,
        maxCombats: maxCombats,
      }

      setMessages((prev) => {
        const updated = [...prev, aiMessage]
        // Save messages to localStorage
        safeLocalStorage.setItem("game_messages", JSON.stringify(updated))
        return updated
      })
      
      // Reset thinking state after successful response
      setIsAiThinking(false)
      
      // Handle game ending
      if (data.is_ending) {
        setInputValue("")  // Clear input
        console.log("Game ended:", data.ending_type)
      }
    } catch (error: any) {
      console.error("Error calling orchestrator:", error)
      setIsAiThinking(false)
      setInputValue("")
      const errorMessage: Message = {
        author: "ai",
        text: `⚠️ Error: ${error?.message || "Failed to process action. Please try again."}`,
        timestamp: Date.now(),
      }
      setMessages((prev) => {
        const updated = [...prev, errorMessage]
        safeLocalStorage.setItem("game_messages", JSON.stringify(updated))
        return updated
      })
    }
  }

  const highlightText = (text: string | undefined) => {
    // Highlight key words with a golden glow effect
    if (!text || typeof text !== 'string') {
      return text || ''
    }
    
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
            {selectedCampaign ? selectedCampaign.replace("-", " ").replace(/\b\w/g, (l) => l.toUpperCase()) : "Classic Dungeon"}
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
              <div className="space-y-3">
                {/* Round and Combat Info - Show from message or current state */}
                {((message.narrationRound !== undefined || currentRound !== undefined) || 
                  (message.combatCount !== undefined || currentCombatCount !== undefined)) && (
                  <div className="ml-8 flex gap-4 text-xs text-gray-500 font-mono">
                    {(message.narrationRound !== undefined || currentRound !== undefined) && (
                      <span>Round: {message.narrationRound !== undefined ? message.narrationRound : currentRound}</span>
                    )}
                    {(message.combatCount !== undefined || currentCombatCount !== undefined) && 
                     (message.maxCombats !== undefined || currentMaxCombats !== undefined) && (
                      <span>Combats: {(message.combatCount !== undefined ? message.combatCount : currentCombatCount) || 0}/{(message.maxCombats !== undefined ? message.maxCombats : currentMaxCombats) || 5}</span>
                    )}
                  </div>
                )}
                
                <div className="flex gap-3">
                  <span className="text-purple-400 font-bold shrink-0">DM:</span>
                  <p
                    className="text-gray-100 leading-relaxed"
                    dangerouslySetInnerHTML={{ __html: highlightText(message.text || '') }}
                  />
                </div>
                
                {/* Ending Message */}
                {message.isEnding && (
                  <div className="ml-8 mt-4 text-center">
                    <div className="text-xl font-bold text-yellow-400 mb-2">
                      {message.endingType === "victory" && "🌟 Adventure Complete! 🌟"}
                      {message.endingType === "defeat" && "💀 Your Journey Ends Here 💀"}
                      {message.endingType === "neutral" && "✨ A New Path Awaits ✨"}
                    </div>
                  </div>
                )}
                
                {/* Display choices if available and not ending */}
                {!message.isEnding && message.choices && message.choices.length > 0 && (
                  <div className="ml-8 space-y-2">
                    <p className="text-gray-400 text-sm font-mono">Choose an action, or type your own:</p>
                    <div className="flex flex-wrap gap-2">
                      {message.choices.map((choice, choiceIndex) => {
                        const isCombat = choice.includes("⚔️") || choice.toLowerCase().includes("combat")
                        const isDisabled = isCombat && choice.includes("Not Available")
                        const isCombatAvailable = isCombat && !isDisabled
                        
                        return (
                          <Button
                            key={choiceIndex}
                            onClick={() => handleChoiceClick(choice)}
                            disabled={isAiThinking || isDisabled || message.isEnding}
                            variant="outline"
                            className={`font-mono text-sm ${
                              isCombatAvailable
                                ? "border-red-500/50 text-red-300 hover:bg-red-500/20 hover:text-red-200 bg-transparent"
                                : isDisabled
                                ? "border-gray-600/30 text-gray-500 bg-transparent cursor-not-allowed"
                                : "border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200 bg-transparent"
                            }`}
                          >
                            {choice}
                          </Button>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex gap-3">
                <span className="text-cyan-400 font-bold shrink-0">You:</span>
                <p className="text-gray-300 leading-relaxed">{message.text || ''}</p>
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
            placeholder={messages.some(m => m.author === "ai" && m.isEnding) ? "Game Over" : "What do you do?"}
            className="flex-1 bg-gray-800 border-gray-600 text-gray-100 placeholder-gray-400 font-mono focus:border-purple-400 focus:ring-purple-400/20"
            disabled={isAiThinking || messages.some(m => m.author === "ai" && m.isEnding)}
          />
          <Button
            type="submit"
            disabled={!inputValue.trim() || isAiThinking || messages.some(m => m.author === "ai" && m.isEnding)}
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
