"""
summarizer_node.py

Graph node that periodically summarizes conversation history and TRIMS the
LangGraph message log. Without trimming, `state.messages` would grow without
bound inside the Postgres checkpointer (every turn appends), inflating row
size, latency, and per-turn LLM token cost.

Runs every N narration rounds: builds a rolling summary into `story_summary`
and emits `RemoveMessage` for everything except the most recent
KEEP_RECENT_MESSAGES so the `add_messages` reducer drops them from state.
"""

import os
import logging
from langchain_core.messages import RemoveMessage
from graph_state import DnDGameState

logger = logging.getLogger(__name__)

# Summarize every N narration rounds
SUMMARIZE_INTERVAL = 5
# Keep the last N messages unsummarized (recent context)
KEEP_RECENT_MESSAGES = 6


def _summarize_with_llm(messages_text: str, existing_summary: str) -> str:
    """Use an LLM to summarize conversation history."""
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback: simple truncation
            return _simple_summarize(messages_text, existing_summary)

        client = OpenAI(api_key=api_key)

        prompt = (
            "You are summarizing a D&D game session for continuity. "
            "Create a concise summary that captures:\n"
            "- Key story events and plot points\n"
            "- Important decisions the player made\n"
            "- Combat outcomes\n"
            "- Current situation and location\n\n"
        )

        if existing_summary:
            prompt += f"Previous summary:\n{existing_summary}\n\n"

        prompt += f"Recent events to incorporate:\n{messages_text}\n\nUpdated summary:"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"LLM summarization failed: {e}")
        return _simple_summarize(messages_text, existing_summary)


def _simple_summarize(messages_text: str, existing_summary: str) -> str:
    """Simple fallback: concatenate and truncate."""
    combined = ""
    if existing_summary:
        combined = existing_summary + "\n\n"
    combined += messages_text

    # Truncate to ~2000 chars, keeping the end (most recent)
    if len(combined) > 2000:
        combined = "..." + combined[-2000:]

    return combined


def summarizer_node(state: DnDGameState) -> dict:
    """
    Summarize older messages into `story_summary` AND emit RemoveMessage tokens
    so the LangGraph `add_messages` reducer drops them from state — keeping the
    checkpointer row size bounded and per-turn LLM cost flat over long sessions.

    Reads: messages, story_summary, narration_round
    Writes: story_summary, messages (as RemoveMessage list to trim history)
    """
    narration_round = state.get("narration_round", 0)
    messages = state.get("messages", [])
    existing_summary = state.get("story_summary", "")

    # Off-interval rounds: only refresh the rolling preview if we don't have one yet.
    if narration_round == 0 or narration_round % SUMMARIZE_INTERVAL != 0:
        if not existing_summary and messages:
            recent = messages[-KEEP_RECENT_MESSAGES:]
            text_parts = []
            for msg in recent:
                content = msg.content if hasattr(msg, "content") else str(msg)
                if content:
                    text_parts.append(content[:300])
            return {"story_summary": "\n\n".join(text_parts)}
        return {}

    if len(messages) <= KEEP_RECENT_MESSAGES:
        return {}

    older_messages = messages[:-KEEP_RECENT_MESSAGES]

    text_parts = []
    for msg in older_messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        if content:
            role = "Player" if hasattr(msg, "type") and msg.type == "human" else "Narrator"
            text_parts.append(f"[{role}] {content[:500]}")

    messages_text = "\n".join(text_parts)
    new_summary = _summarize_with_llm(messages_text, existing_summary)

    # Drop summarized messages from state so history stays bounded. Each
    # message must have an `id` for RemoveMessage to take effect; LangChain
    # assigns one automatically when messages are appended via add_messages.
    removals = [
        RemoveMessage(id=msg.id)
        for msg in older_messages
        if getattr(msg, "id", None)
    ]

    logger.info(
        "Summarized %d messages into %d chars; trimming %d from state",
        len(older_messages),
        len(new_summary),
        len(removals),
    )

    return {
        "story_summary": new_summary,
        "messages": removals,
    }
