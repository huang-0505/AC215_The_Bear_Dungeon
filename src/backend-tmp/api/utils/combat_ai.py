"""
combat_ai.py
AI components for DnD combat: action parsing, enemy AI, and narrative generation.
Uses Google GenAI for LLM-based decision making and narration.
"""

import os
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional
from google import genai
from google.genai import types

from api.utils.combat_engine import Character, CombatEngine, ACTION_REGISTRY
from api.utils.db_tool import retrieve_top_k

# Thread pool for running blocking GenAI calls
_genai_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="genai")


# ========== Action Parser for Players ==========
class ActionParser:
    """Parses natural language player input into structured actions using semantic search."""

    def __init__(self, engine: CombatEngine):
        self.engine = engine
        self.action_db_path = 'data/embeddings-actions.jsonl'
        self.enemy_db_path = 'data/embeddings-enemies.jsonl'

    def parse(self, actor: Character, user_input: str) -> Optional[dict]:
        """
        Parse player input into action data.

        Args:
            actor: The acting character
            user_input: Natural language action description

        Returns:
            Dict with action_id, type, and target, or None if parsing fails
        """
        # Find closest matching action
        try:
            action_id = retrieve_top_k(user_input, self.action_db_path, k=1)[0]
        except:
            # Fallback to melee attack if database not available
            action_id = 0

        action_type = ACTION_REGISTRY[action_id]

        # Find closest matching target among alive enemies
        alive_enemies = self.engine.state.get_alive(role="enemy")
        if not alive_enemies:
            return None

        try:
            enemy_ids = retrieve_top_k(user_input, self.enemy_db_path, k=100)
            # Match retrieved IDs with alive enemies
            target = None
            for enemy_id in enemy_ids:
                if any(e.id == enemy_id for e in alive_enemies):
                    target = self.engine.state.get_by_id(enemy_id, role="enemy")
                    break

            if not target:
                target = alive_enemies[0]  # Default to first alive enemy
        except:
            # Fallback to first alive enemy if database not available
            target = alive_enemies[0]

        return {
            "id": action_id,
            "type": action_type,
            "target": target
        }


# ========== Action Parser for Enemy AI ==========
class ActionParserBot:
    """Parses enemy AI decisions into structured actions."""

    def __init__(self, engine: CombatEngine):
        self.engine = engine
        self.action_db_path = 'data/embeddings-actions.jsonl'
        self.ally_db_path = 'data/embeddings-allies.jsonl'

    def parse(self, actor: Character, action_text: str) -> Optional[dict]:
        """
        Parse bot-generated action description into action data.
        Handles both enemies (targeting players/teammates) and teammates (targeting enemies).

        Args:
            actor: The acting character (enemy or teammate)
            action_text: LLM-generated action description

        Returns:
            Dict with action_id, type, and target, or None if parsing fails
        """
        # Find closest matching action
        try:
            action_id = retrieve_top_k(action_text, self.action_db_path, k=1)[0]
        except:
            action_id = 0  # Default to melee attack

        action_type = ACTION_REGISTRY[action_id]

        # Determine valid targets based on actor role
        if actor.role == "teammate":
            # Teammates target enemies
            valid_targets = self.engine.state.get_alive(role="enemy")
            target_role = "enemy"
        else:
            # Enemies target players and teammates
            valid_targets = self.engine.state.get_alive(role="player") + self.engine.state.get_alive(role="teammate")
            target_role = "player"  # For ID lookup, but we'll search both roles

        if not valid_targets:
            return None

        # Try to find target using semantic search
        try:
            ally_ids = retrieve_top_k(action_text, self.ally_db_path, k=100)
            target = None
            # Match retrieved IDs with valid targets
            for ally_id in ally_ids:
                # Check players first
                if target_role == "player":
                    if any(p.id == ally_id for p in self.engine.state.get_alive(role="player")):
                        target = self.engine.state.get_by_id(ally_id, role="player")
                        break
                    # Also check teammates
                    if any(t.id == ally_id for t in self.engine.state.get_alive(role="teammate")):
                        target = self.engine.state.get_by_id(ally_id, role="teammate")
                        break
                else:
                    # For enemies, check enemy list
                    if any(e.id == ally_id for e in self.engine.state.get_alive(role="enemy")):
                        target = self.engine.state.get_by_id(ally_id, role="enemy")
                        break

            # Fallback: try name matching if semantic search didn't work
            if not target:
                action_lower = action_text.lower()
                for t in valid_targets:
                    if t.name.lower() in action_lower:
                        target = t
                        break

            # Final fallback: use tactical or random target selection instead of always first
            if not target:
                # Use tactical selection: prioritize low HP and lower AC targets
                if len(valid_targets) == 1:
                    target = valid_targets[0]
                else:
                    # Score targets: lower HP% and lower AC = better target
                    scored = []
                    for t in valid_targets:
                        hp_percent = t.hp / t.max_hp if t.max_hp > 0 else 1.0
                        ac_factor = (20 - t.ac) / 10.0  # Lower AC = easier to hit
                        score = (hp_percent * 0.7) - (ac_factor * 0.3) + (random.random() * 0.1)
                        scored.append((t, score))
                    # Sort by score (lower = better), pick from top 2-3 for variety
                    scored.sort(key=lambda x: x[1])
                    top_n = min(3, len(scored))
                    target = random.choice(scored[:top_n])[0]
        except:
            # Fallback to tactical/random target if database not available
            if len(valid_targets) == 1:
                target = valid_targets[0]
            else:
                # Use random selection for variety
                target = random.choice(valid_targets)

        return {
            "id": action_id,
            "type": action_type,
            "target": target
        }


# ========== Enemy AI Agent ==========
class DnDBot:
    """AI agent for controlling enemy actions using LLM reasoning."""

    def __init__(self, engine: CombatEngine, model: str = "gemini-2.0-flash-001"):
        self.engine = engine
        self.model = model
        # Initialize Google GenAI client using GCP credentials (if available)
        gcp_project = os.environ.get("GCP_PROJECT")
        gcp_location = os.environ.get("GCP_LOCATION", "us-central1")
        self.client = None
        self.use_llm = False
        print(f"[DnDBot] Initializing with GCP_PROJECT={gcp_project}, GCP_LOCATION={gcp_location}")
        try:
            if gcp_project:
                print(f"[DnDBot] Attempting to initialize GenAI client...")
                self.client = genai.Client(
                    vertexai=True,
                    project=gcp_project,
                    location=gcp_location
                )
                self.use_llm = True
                print(f"[DnDBot] GenAI client initialized successfully, use_llm=True")
            else:
                print("[DnDBot] GCP_PROJECT not set, enemy bot will use simple attack strategy")
        except Exception as e:
            print(f"[DnDBot] Failed to initialize GenAI client: {e}, using fallback strategy")
            import traceback
            traceback.print_exc()
        self.parser = ActionParserBot(engine)

    def _call_genai_sync(self, prompt: str) -> Optional[str]:
        """Synchronous GenAI call (runs in thread pool)"""
        if not self.client:
            print(f"[DnDBot._call_genai_sync] ERROR: Client is None, cannot make GenAI call")
            return None
        try:
            print(f"[DnDBot._call_genai_sync] Making GenAI API call...")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=150,
                    temperature=0.7,
                )
            )
            result = response.text if response and response.text else None
            print(f"[DnDBot._call_genai_sync] GenAI API call completed, result length: {len(result) if result else 0}")
            return result
        except Exception as e:
            print(f"[DnDBot._call_genai_sync] GenAI call error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _select_tactical_target(self, targets: list, actor: Character) -> Optional[Character]:
        """
        Select a tactical target using smart heuristics.
        Prioritizes: low HP (finish off wounded), lower AC (easier to hit), with some randomness.
        
        Args:
            targets: List of potential target characters
            actor: The acting character
            
        Returns:
            Selected target character or None if no targets
        """
        if not targets:
            return None
        
        if len(targets) == 1:
            return targets[0]
        
        # Score each target based on tactical factors
        scored_targets = []
        for t in targets:
            # Calculate HP percentage (lower is better - finish off wounded enemies)
            hp_percent = t.hp / t.max_hp if t.max_hp > 0 else 1.0
            
            # AC factor (lower AC is easier to hit, so better target)
            # Normalize AC to 0-1 scale (assuming AC range 10-20, lower is better)
            ac_factor = (20 - t.ac) / 10.0  # Higher value = easier to hit
            
            # Tactical score: prioritize low HP (70% weight) and lower AC (30% weight)
            # Add small random factor (10%) to avoid being too predictable
            tactical_score = (hp_percent * 0.7) - (ac_factor * 0.3) + (random.random() * 0.1)
            
            scored_targets.append((t, tactical_score, hp_percent, t.ac))
        
        # Sort by tactical score (lower score = better target)
        scored_targets.sort(key=lambda x: x[1])
        
        # Pick from top 2-3 candidates to add some variety
        top_n = min(3, len(scored_targets))
        selected = random.choice(scored_targets[:top_n])[0]
        
        print(f"[DnDBot] Tactical target selection: {selected.name} (HP: {selected.hp}/{selected.max_hp}, AC: {selected.ac})")
        return selected

    async def decide_action(self) -> Optional[dict]:
        """
        Generate enemy action using LLM tactical reasoning.
        Falls back to tactical attack if LLM fails or times out.
        Uses async with timeout to prevent freezing.

        Returns:
            Action data dict or None if decision fails
        """
        print(f"[DnDBot] decide_action() called")
        actor = self.engine.current_actor
        if not actor:
            print(f"[DnDBot] ERROR: No current actor")
            return None

        # Determine targets based on actor role
        if actor.role == "teammate":
            # Teammates target enemies
            targets = self.engine.state.get_alive(role="enemy")
            allies = self.engine.state.get_alive(role="player") + self.engine.state.get_alive(role="teammate")
            allies = [a for a in allies if a != actor]  # Exclude self
            enemies = self.engine.state.get_alive(role="enemy")
            bot_type = "teammate"
        else:
            # Enemies target players/teammates
            targets = self.engine.state.get_alive(role="player") + self.engine.state.get_alive(role="teammate")
            allies = self.engine.state.get_alive(role="enemy")
            allies = [a for a in allies if a != actor]  # Exclude self
            enemies = self.engine.state.get_alive(role="player") + self.engine.state.get_alive(role="teammate")
            bot_type = "enemy"

        if not targets:
            print(f"[DnDBot] ERROR: No alive targets for {actor.role}")
            return None
        
        # Use tactical target selection instead of random
        target = self._select_tactical_target(targets, actor)
        if not target:
            print(f"[DnDBot] ERROR: Failed to select tactical target")
            return None
        print(f"[DnDBot] Selected tactical target: {target.name} (actor role: {actor.role})")
        
        # Try to use LLM for more interesting actions (only if client is available)
        if self.use_llm and self.client:
            try:
                # Build detailed target information for tactical decision-making
                target_info = []
                for t in targets:
                    hp_percent = int((t.hp / t.max_hp) * 100) if t.max_hp > 0 else 100
                    status = "CRITICAL" if hp_percent < 25 else "WOUNDED" if hp_percent < 50 else "HEALTHY"
                    target_info.append(f"  - {t.name}: HP {t.hp}/{t.max_hp} ({hp_percent}%, {status}), AC {t.ac}")
                targets_str = "\n".join(target_info)
                
                # Build ally information
                ally_info = []
                for a in allies:
                    hp_percent = int((a.hp / a.max_hp) * 100) if a.max_hp > 0 else 100
                    ally_info.append(f"  - {a.name}: HP {a.hp}/{a.max_hp} ({hp_percent}%), AC {a.ac}")
                allies_str = "\n".join(ally_info) if ally_info else "  (none)"
                
                # Construct tactical analysis prompt with detailed combat state
                if actor.role == "teammate":
                    analysis_prompt = f"""
You are a DnD teammate bot controlling {actor.name}, fighting alongside your party.

COMBAT STATE:
- You: {actor.name} (HP: {actor.hp}/{actor.max_hp}, AC: {actor.ac})

- Your Allies:
{allies_str}

- Enemy Targets (choose one to attack):
{targets_str}

TACTICAL GUIDANCE:
- Prioritize finishing off wounded enemies (low HP) to reduce enemy numbers
- Consider targeting enemies with lower AC (easier to hit)
- Focus fire on one enemy if multiple are wounded
- Protect your allies if they're in critical condition

Choose one action to help your party defeat the enemies. Describe who you attack (or target) and how you perform it.
Stay consistent with D&D combat style and the character's role.
Keep your response concise (1-2 sentences).
Example: "I attack {target.name} with my weapon" or "I focus my attack on the wounded {target.name}"
"""
                else:
                    analysis_prompt = f"""
You are a DnD enemy bot controlling {actor.name}.

COMBAT STATE:
- You: {actor.name} (HP: {actor.hp}/{actor.max_hp}, AC: {actor.ac})

- Your Allies:
{allies_str}

- Enemy Targets (choose one to attack):
{targets_str}

TACTICAL GUIDANCE:
- Prioritize finishing off wounded enemies (low HP) to eliminate threats quickly
- Consider targeting enemies with lower AC (easier to hit)
- Focus fire on one enemy if multiple are wounded
- Target spellcasters or support characters if they're vulnerable

Choose one hostile action and describe it in natural language.
Describe who you attack (or target) and how you perform it.
Stay consistent with D&D combat style and the character's role.
Keep your response concise (1-2 sentences).
Example: "I attack {target.name} with my weapon" or "I focus my attack on the wounded {target.name}"
"""

                # Run GenAI call in thread pool with 5 second timeout
                # Use get_running_loop() for better async compatibility (Python 3.7+)
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # Fallback if no running loop (shouldn't happen in FastAPI)
                    loop = asyncio.get_event_loop()
                
                print(f"[DnDBot] Starting GenAI call for {actor.name} (timeout: 3s)")
                try:
                    # Use a shorter timeout (3 seconds) to fail fast
                    action_text = await asyncio.wait_for(
                        loop.run_in_executor(_genai_executor, self._call_genai_sync, analysis_prompt),
                        timeout=3.0
                    )
                    print(f"[DnDBot] GenAI call completed for {actor.name}")
                    
                    if action_text:
                        # Parse LLM output into structured action
                        action = self.parser.parse(actor, action_text)
                        if action:
                            print(f"[DnDBot] Successfully parsed AI action: {action}")
                            return action
                        else:
                            print(f"[DnDBot] Failed to parse AI response, using fallback")
                    else:
                        print(f"[DnDBot] Empty AI response, using fallback")
                except asyncio.TimeoutError:
                    print(f"[DnDBot] GenAI call timed out after 3 seconds for {actor.name}, using fallback")
                except Exception as e:
                    print(f"[DnDBot] Error in async AI decision for {actor.name}: {e}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"[DnDBot] Error setting up AI decision (using fallback): {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[DnDBot] GenAI not available (use_llm={self.use_llm}, client={self.client is not None}), using fallback")
        
        # Fallback to simple attack (always works, never hangs)
        fallback_action = {
            "id": 0,
            "type": "MeleeAttack",
            "target": target
        }
        print(f"[DnDBot] Returning fallback action: {fallback_action}")
        return fallback_action


class DnDNarrator:
    """AI narrator for generating vivid combat descriptions"""

    def __init__(self, model: str = "gemini-2.0-flash-001"):
        self.model = model
        gcp_project = os.environ.get("GCP_PROJECT")
        gcp_location = os.environ.get("GCP_LOCATION", "us-central1")
        self.client = None
        self.use_genai = False
        try:
            if gcp_project:
                print(f"[DnDNarrator] Initializing GenAI client...")
                self.client = genai.Client(
                    vertexai=True,
                    project=gcp_project,
                    location=gcp_location
                )
                self.use_genai = True
                print(f"[DnDNarrator] GenAI client initialized successfully")
            else:
                print(f"[DnDNarrator] GCP_PROJECT not set, using fallback narration")
        except Exception as e:
            print(f"[DnDNarrator] Failed to initialize GenAI client: {e}, using fallback narration")
            self.client = None
            self.use_genai = False

    def _call_genai_sync(self, prompt: str) -> Optional[str]:
        """Synchronous GenAI call for narration (runs in thread pool)"""
        if not self.client:
            return None
        try:
            print(f"[DnDNarrator._call_genai_sync] Making GenAI API call...")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=200,
                    temperature=0.8,
                )
            )
            result = response.text.strip() if response and response.text else None
            print(f"[DnDNarrator._call_genai_sync] GenAI API call completed, result length: {len(result) if result else 0}")
            return result
        except Exception as e:
            print(f"[DnDNarrator._call_genai_sync] GenAI call error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def narrate(self, user_query: str, action_result: str) -> str:
        """
        Generate vivid narrative combining player intent and mechanical result.
        Uses async with timeout to prevent blocking.

        Args:
            user_query: Player's action description
            action_result: Mechanical outcome from engine

        Returns:
            Dramatic narrative description
        """
        # If GenAI is not available, return simple narration immediately
        if not self.use_genai or not self.client:
            print(f"[DnDNarrator] GenAI not available, using fallback narration")
            return action_result

        narrative_prompt = f"""
You are a DnD Narrator describing an epic battle scene.

Player's Intent: "{user_query}"
Actual Result: "{action_result}"

Create a vivid, dramatic description (2-3 sentences) that:
- Describes the action cinematically
- Incorporates the player's intended action style
- Shows the actual mechanical outcome
- Uses D&D-style evocative language

IMPORTANT: When user_query and action_result conflict, prioritize the PLAYER'S INTENT in your narrative style while still showing the actual outcome.

Be dramatic and immersive. Make it feel epic!
"""

        try:
            print(f"[DnDNarrator] Generating narrative...")
            loop = asyncio.get_running_loop()
            try:
                narrative_text = await asyncio.wait_for(
                    loop.run_in_executor(_genai_executor, self._call_genai_sync, narrative_prompt),
                    timeout=5.0  # 5 second timeout for narration
                )
                if narrative_text:
                    print(f"[DnDNarrator] Narrative generated successfully")
                    return narrative_text
                else:
                    print(f"[DnDNarrator] Empty narrative response, using fallback")
                    return action_result
            except asyncio.TimeoutError:
                print(f"[DnDNarrator] Narrative generation timed out after 5 seconds, using fallback")
                return action_result
            except Exception as e:
                print(f"[DnDNarrator] Error in async narration: {e}")
                import traceback
                traceback.print_exc()
                return action_result
        except Exception as e:
            print(f"[DnDNarrator] Error setting up narration (using fallback): {e}")
            import traceback
            traceback.print_exc()
            return action_result
