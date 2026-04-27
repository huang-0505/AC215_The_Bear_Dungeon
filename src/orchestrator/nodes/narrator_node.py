"""
narrator_node.py

Graph node that calls the finetuned Gemini narrator agent
and handles story tree navigation.
"""

import os
import re
import logging
import threading
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage

from graph_state import DnDGameState
from story_tree_loader import StoryTreeLoader, StoryTree

logger = logging.getLogger(__name__)

# --- Narrator LLM setup (initialized lazily, thread-safe) ---
_llm_client = None
_narrator_endpoint = None
_narrator_config = None
_openai_client = None
_llm_lock = threading.Lock()
_openai_lock = threading.Lock()


def _get_llm_client() -> tuple:
    """Lazily initialize the Gemini client for narrator (thread-safe)."""
    global _llm_client, _narrator_endpoint, _narrator_config
    if _llm_client is not None:
        return _llm_client, _narrator_endpoint, _narrator_config

    with _llm_lock:
        if _llm_client is not None:
            return _llm_client, _narrator_endpoint, _narrator_config

        from google import genai
        from google.genai import types

        gcp_project = os.getenv("GCP_PROJECT")
        gcp_location = os.getenv("GCP_LOCATION", "us-central1")
        narrator_endpoint_id = os.getenv("NARRATOR_ENDPOINT_ID", "5165249441082376192")

        if not gcp_project:
            logger.warning("GCP_PROJECT not set. Narrator features may not work.")
            return None, None, None

        _narrator_endpoint = f"projects/{gcp_project}/locations/{gcp_location}/endpoints/{narrator_endpoint_id}"
        _narrator_config = types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=0.8,
            top_p=0.95,
        )
        try:
            _llm_client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
        except Exception as e:
            logger.error(f"Failed to initialize GenAI client: {e}")
            return None, None, None

    return _llm_client, _narrator_endpoint, _narrator_config


def _get_openai_client():
    """Lazily initialize OpenAI client for combat trigger detection (thread-safe)."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    with _openai_lock:
        if _openai_client is not None:
            return _openai_client

        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _openai_client = OpenAI(api_key=api_key)

    return _openai_client


def _build_narrator_prompt(
    user_input: str,
    rules_context: Optional[str] = None,
    story_context: Optional[str] = None,
    generate_choices: bool = True,
) -> str:
    """Build the prompt for the narrator (shared by Gemini and OpenAI fallback)."""
    prompt_parts = []
    if story_context:
        prompt_parts.append(f"Story Context (what happened before in this adventure):\n{story_context}\n")
    prompt_parts.append(f"Player action: {user_input}\n\nNarrate the outcome:")
    if rules_context:
        prompt_parts.insert(-1, f"Relevant D&D rules:\n{rules_context}\n")
    if generate_choices:
        prompt_parts.append(
            "\n\nAfter your narration, provide 3-4 suggested action choices for the player. "
            "Format choices as:\nCHOICES:\n1. [choice 1]\n2. [choice 2]\n3. [choice 3]"
        )
    return "\n".join(prompt_parts)


def _call_narrator_openai(prompt: str) -> dict:
    """Fallback narrator using OpenAI GPT-4o-mini when Vertex AI is unavailable."""
    client = _get_openai_client()
    if not client:
        return {
            "result": "The narrator is temporarily unavailable. Please try again.",
            "choices": None,
        }

    system_prompt = (
        "You are a masterful D&D Dungeon Master narrating an immersive fantasy adventure. "
        "Write vivid, atmospheric narration in second person (\"you\"). "
        "Keep responses to 2-3 paragraphs. Be creative and dramatic. "
        "Always provide unique, contextually appropriate choices based on the current situation."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=1000,
        )
        result = response.choices[0].message.content
        choices = _extract_choices(result)
        if choices:
            result = _remove_choices_section(result)
        return {"result": result, "choices": choices}

    except Exception as e:
        logger.error(f"OpenAI narrator fallback failed: {e}")
        return {"result": f"Narrator error. Please try again.", "choices": None}


def _call_narrator(
    user_input: str,
    rules_context: Optional[str] = None,
    story_context: Optional[str] = None,
    generate_choices: bool = True,
) -> dict:
    """Call the narrator model. Tries Vertex AI first, falls back to OpenAI."""
    prompt = _build_narrator_prompt(user_input, rules_context, story_context, generate_choices)

    # Try Vertex AI finetuned narrator first
    client, endpoint, config = _get_llm_client()

    if client and endpoint:
        try:
            response = client.models.generate_content(
                model=endpoint, contents=prompt, config=config,
            )

            result = None
            if hasattr(response, "text") and response.text:
                result = response.text
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
                    text_parts = [p.text for p in candidate.content.parts if hasattr(p, "text") and p.text]
                    if text_parts:
                        result = " ".join(text_parts)

            if result:
                choices = _extract_choices(result)
                if choices:
                    result = _remove_choices_section(result)
                return {"result": result, "choices": choices}

            logger.warning("Vertex AI narrator returned empty response, falling back to OpenAI")

        except Exception as e:
            logger.warning(f"Vertex AI narrator failed ({e}), falling back to OpenAI")

    # Fallback to OpenAI
    logger.info("Using OpenAI fallback narrator")
    return _call_narrator_openai(prompt)


def _extract_choices(text: str) -> Optional[list[str]]:
    """Extract choice options from narrator response text."""
    choices_match = re.search(
        r"CHOICES:\s*\n((?:\d+\.\s*[^\n]+\n?)+)", text, re.IGNORECASE | re.MULTILINE,
    )
    if choices_match:
        choices_text = choices_match.group(1)
        choices = re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|\n*$)", choices_text, re.MULTILINE)
        choices = [c.strip() for c in choices if c.strip()]
        if choices:
            return choices

    # Fallback: use OpenAI to extract choices
    client = _get_openai_client()
    if not client:
        return None

    try:
        import json
        prompt = (
            f"Extract 3-4 suggested action choices from this D&D narrative text. "
            f"Return a JSON object with a \"choices\" key containing an array of strings.\n\n"
            f"Text: {text}\n\n"
            f"Example format: {{\"choices\": [\"Investigate the door\", \"Search for traps\", \"Try to pick the lock\"]}}"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        if isinstance(parsed, list):
            return parsed[:4]
        if "choices" in parsed and isinstance(parsed["choices"], list):
            return parsed["choices"][:4]
    except Exception as e:
        logger.warning(f"Failed to extract choices with LLM: {e}")

    return None


def _remove_choices_section(text: str) -> str:
    """Remove the CHOICES section from narrative text."""
    return re.sub(
        r"CHOICES:\s*\n((?:\d+\.\s*[^\n]+\n?)+)", "", text, flags=re.IGNORECASE | re.MULTILINE,
    ).strip()


def _detect_combat_trigger(text: str) -> bool:
    """Use LLM to detect if narrative indicates combat start."""
    client = _get_openai_client()
    if not client:
        return False

    prompt = (
        f"Analyze this D&D narrative text and determine if it describes the START of a combat encounter.\n"
        f"Look for phrases like: \"enemies appear\", \"ambush\", \"attack\", \"roll for initiative\",\n"
        f"monster/enemy descriptions appearing, hostile NPCs engaging, \"You are attacked\".\n\n"
        f"Text: \"{text}\"\n\nAnswer with only \"YES\" or \"NO\"."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return "yes" in response.choices[0].message.content.lower()
    except Exception as e:
        logger.error(f"Error detecting combat trigger: {e}")
        return False


def _load_story_tree(campaign_id: Optional[str]) -> Optional[StoryTree]:
    """Load story tree from campaign ID."""
    if not campaign_id:
        return None
    return StoryTreeLoader.load_story_tree(campaign_id)


def generate_initial_choices(prompt: str) -> list[str]:
    """
    Public API: Generate initial choices for a game start prompt.
    Used by app.py to seed the first turn with choices.
    """
    result = _call_narrator(prompt, generate_choices=True)
    return result.get("choices") or []


def narrator_node(state: DnDGameState) -> dict:
    """
    Handle narration: navigate story tree or call AI narrator.
    Increments narration round. Detects AI-based combat triggers.

    Reads: player_action, validation_result, campaign_id, current_story_node_id,
           story_summary, narration_round
    Writes: narrator_response, choices, narration_round, state_type, is_ending,
            ending_type, current_story_node_id, combat_trigger, messages
    """
    player_action = state["player_action"]
    validation = state.get("validation_result", {})
    rules_context = validation.get("rule_text")
    campaign_id = state.get("campaign_id")
    current_story_node_id = state.get("current_story_node_id")
    story_summary = state.get("story_summary", "")

    # Increment narration round
    narration_round = state.get("narration_round", 0) + 1

    # Try story tree navigation first
    response_text = None
    response_choices = None
    is_ending = False
    ending_type = None
    next_story_node_id = current_story_node_id
    new_state_type = "narration"

    story_tree = _load_story_tree(campaign_id)

    if story_tree and current_story_node_id:
        story_node = story_tree.get_node(current_story_node_id)
        if story_node:
            if story_node.is_ending:
                is_ending = True
                ending_type = story_node.ending_type
                response_text = story_node.narrative
                response_choices = []
                new_state_type = "game_over"
            else:
                next_node = story_tree.get_next_node_for_choice(current_story_node_id, player_action)
                if next_node:
                    response_text = next_node.narrative
                    response_choices = next_node.choices
                    next_story_node_id = next_node.node_id
                    is_ending = next_node.is_ending
                    ending_type = next_node.ending_type
                    if is_ending:
                        new_state_type = "game_over"

    # Fall back to AI narrator if no story tree match
    if not response_text:
        narrator_result = _call_narrator(
            player_action,
            rules_context=rules_context,
            story_context=story_summary,
            generate_choices=True,
        )
        response_text = narrator_result["result"]
        response_choices = narrator_result.get("choices")

    # Ensure we always have choices
    if not response_choices:
        response_choices = [
            "Continue exploring",
            "Investigate the area",
            "Search for clues",
        ]

    # Check for AI-detected combat trigger
    combat_trigger = "none"
    if not is_ending and _detect_combat_trigger(response_text):
        logger.info("Combat triggered by AI detection during narration")
        combat_trigger = "ai_detected"

    # Append to conversation history
    new_messages = [
        HumanMessage(content=player_action),
        AIMessage(content=response_text),
    ]

    return {
        "narrator_response": response_text,
        "choices": response_choices,
        "narration_round": narration_round,
        "state_type": new_state_type,
        "is_ending": is_ending,
        "ending_type": ending_type,
        "current_story_node_id": next_story_node_id,
        "combat_trigger": combat_trigger,
        "response": response_text,
        "messages": new_messages,
    }
