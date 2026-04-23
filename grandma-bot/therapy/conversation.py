"""Core conversation engine — generates Claude-powered therapy responses.

Three public coroutines:
  generate_therapy_response(...)  — main per-turn Claude call
  generate_opener(...)            — first message in a session (template or Claude)
  generate_safety_response(...)   — immediate redirect for distress situations (sync)
"""
from __future__ import annotations

import logging
import random
from typing import Optional

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import settings
from .prompts import (
    OPENER_TEMPLATES,
    PHASE_INSTRUCTIONS,
    SAFETY_RESPONSES,
    SYSTEM_PROMPT_TEMPLATE,
)
from .state_machine import SessionPhase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 150
TEMPERATURE = 0.8
NOVEL_OPENER_PROBABILITY = 0.30  # fraction of sessions that get a Claude-generated opener

_RETRY_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

# ---------------------------------------------------------------------------
# Shared async client (one per process, reused across requests)
# ---------------------------------------------------------------------------

_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_profile_facts(profile_facts: list[dict]) -> str:
    if not profile_facts:
        return "Nothing recorded yet — this is our first conversation."
    lines = [f"- {f['fact']}" for f in profile_facts]
    return "\n".join(lines)


def _format_tags(tags: list | None) -> str:
    if not tags:
        return "none"
    return ", ".join(str(t) for t in tags)


def _build_system_prompt(
    grandma_name: str,
    memory: dict,
    profile_facts: list[dict],
    current_phase: SessionPhase,
) -> str:
    phase_instr_template = PHASE_INSTRUCTIONS.get(
        current_phase,
        PHASE_INSTRUCTIONS[SessionPhase.REFLECT],
    )
    # Phase instructions reference {grandma_name} in some variants.
    phase_instructions = phase_instr_template.format(grandma_name=grandma_name)

    return SYSTEM_PROMPT_TEMPLATE.format(
        grandma_name=grandma_name,
        memory_summary=memory.get("ai_summary") or memory.get("original_caption") or "a cherished family photo",
        memory_tags=_format_tags(memory.get("ai_tags")),
        people=_format_tags(memory.get("people_mentioned")),
        emotion_hints=_format_tags(memory.get("emotion_hints")),
        profile_facts=_format_profile_facts(profile_facts),
        phase_instructions=phase_instructions,
    )


def _build_transcript(
    grandma_name: str,
    session_turns: list[dict],
    last_grandma_message: str,
) -> str:
    """Format prior turns as a readable transcript ending with grandma's latest message."""
    lines: list[str] = []
    for turn in session_turns:
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        speaker = "You" if role == "bot" else grandma_name
        lines.append(f"{speaker}: {content}")

    # Append the current inbound message (may not be in DB yet).
    if last_grandma_message.strip():
        lines.append(f"{grandma_name}: {last_grandma_message.strip()}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retry wrapper for Claude calls
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type(_RETRY_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _call_claude(system: str, user: str) -> str:
    """Make a single Claude API call and return the text response."""
    response = await _get_client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(
        block.text for block in response.content if hasattr(block, "text")
    ).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_therapy_response(
    memory: dict,
    session_turns: list[dict],
    profile_facts: list[dict],
    grandma_name: str,
    current_phase: SessionPhase,
    last_grandma_message: str,
) -> str:
    """Generate the bot's next message using Claude.

    Args:
        memory:               Row from the memories table.
        session_turns:        All turns recorded so far for this session.
        profile_facts:        Rows from grandma_profile_facts for this grandma.
        grandma_name:         First name used throughout the conversation.
        current_phase:        Current SessionPhase (drives phase_instructions).
        last_grandma_message: The message we are responding to (may not be in
                              session_turns yet).

    Returns:
        The bot's next message, ready to send over iMessage.
    """
    system_prompt = _build_system_prompt(grandma_name, memory, profile_facts, current_phase)
    transcript = _build_transcript(grandma_name, session_turns, last_grandma_message)

    logger.debug(
        "[conversation] generate_therapy_response | phase=%s\n"
        "--- SYSTEM ---\n%s\n--- USER ---\n%s",
        current_phase.value,
        system_prompt,
        transcript,
    )

    try:
        reply = await _call_claude(system_prompt, transcript)
    except Exception as exc:
        logger.error("[conversation] Claude API failed after retries: %s", exc)
        # Graceful fallback — generic warm continuation.
        reply = "That sounds so lovely. Tell me more? 💛"

    logger.debug("[conversation] Claude reply: %r", reply)
    return reply


async def generate_opener(
    memory: dict,
    grandma_name: str,
    profile_facts: list[dict],
) -> str:
    """Generate the very first message of a session (GREET_ANCHOR phase).

    70 % of the time: fill a random hand-crafted template (fast, reliable).
    30 % of the time: ask Claude for a novel opener (adds variety).

    Args:
        memory:       Row from the memories table.
        grandma_name: Grandma's first name.
        profile_facts: Known facts (used only for the Claude path).

    Returns:
        The opening iMessage text to send.
    """
    memory_summary = (
        memory.get("ai_summary") or memory.get("original_caption") or "a cherished family photo"
    )

    if random.random() >= NOVEL_OPENER_PROBABILITY:
        # Template path (fast, always works)
        template = random.choice(OPENER_TEMPLATES)
        opener = template.format(grandma_name=grandma_name, memory_summary=memory_summary)
        logger.debug("[conversation] opener via template: %r", opener)
        return opener

    # Claude path — generate a novel opener
    system = (
        f"You are writing the very first iMessage to {grandma_name}, an elderly woman "
        "whose family has just shared a cherished photo with you. "
        "Write ONE warm, short opening message (3 sentences max) that: "
        "(1) greets her by name, (2) briefly describes the photo, "
        "(3) asks a single gentle question inviting her to remember. "
        "Use loving, everyday language. One emoji is okay. "
        "No narration, no labels — just the message itself."
    )
    user = (
        f"Photo description: {memory_summary}\n"
        f"Tags: {_format_tags(memory.get('ai_tags'))}\n"
        f"People: {_format_tags(memory.get('people_mentioned'))}"
    )

    logger.debug("[conversation] generating novel opener via Claude")
    try:
        opener = await _call_claude(system, user)
    except Exception as exc:
        logger.error("[conversation] novel opener failed, falling back to template: %s", exc)
        template = random.choice(OPENER_TEMPLATES)
        opener = template.format(grandma_name=grandma_name, memory_summary=memory_summary)

    logger.debug("[conversation] opener via Claude: %r", opener)
    return opener


def generate_safety_response(grandma_name: str) -> str:  # noqa: ARG001 — reserved for future personalisation
    """Return a pre-written, non-clinical redirect for distress situations.

    Deliberately synchronous and never calls Claude — safety responses must
    be instantaneous and 100 % reliable.
    """
    return random.choice(SAFETY_RESPONSES)
