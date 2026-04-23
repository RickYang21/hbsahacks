"""Profile fact extraction — learns new things about grandma from each message.

Public API
----------
extract_facts(...)          — async; calls Claude to pull new facts from a message
save_extracted_facts(...)   — async; writes confirmed facts to grandma_profile_facts
deduplicate_facts(...)      — pure; removes new facts that overlap existing ones

Background usage (non-blocking)
--------------------------------
    import asyncio
    asyncio.create_task(
        _extract_and_save(grandma_id, session_id, message, memory, history, existing)
    )
"""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import settings
from .conversation import _get_client  # reuse the shared async client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXTRACTION_MODEL = "claude-haiku-4-5-20251001"  # fast + cheap for background task
_MIN_CONFIDENCE = 0.4
# Stop words excluded when comparing facts for similarity
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "that", "this", "it", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "i", "my",
    "me", "she", "her", "he", "his", "they", "their", "we", "our", "you",
    "your", "so", "as", "up", "out", "about", "into", "than", "more",
})
# Two facts sharing this many significant words are considered duplicates.
_OVERLAP_THRESHOLD = 3

_RETRY_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _significant_words(text: str) -> set[str]:
    """Return lowercase non-stop words from *text*."""
    words = text.lower().split()
    # Strip punctuation from each word before stop-word filtering.
    cleaned = [w.strip(".,!?;:\"'()[]") for w in words]
    return {w for w in cleaned if w and w not in _STOP_WORDS}


def _format_history(history: list[dict]) -> str:
    """Render the last ≤6 turns of conversation history as plain text."""
    recent = history[-6:]
    lines: list[str] = []
    for turn in recent:
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        speaker = "Companion" if role == "bot" else "Grandma"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _format_existing_facts(existing_facts: list[str]) -> str:
    if not existing_facts:
        return "None recorded yet."
    return "\n".join(f"- {f}" for f in existing_facts)


# ---------------------------------------------------------------------------
# Core extraction call
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type(_RETRY_ERRORS),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
)
async def _call_extraction(system: str, user: str) -> str:
    response = await _get_client().messages.create(
        model=_EXTRACTION_MODEL,
        max_tokens=512,
        temperature=0.2,  # low temperature → consistent, factual output
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(
        block.text for block in response.content if hasattr(block, "text")
    ).strip()


# ---------------------------------------------------------------------------
# Public: extract_facts
# ---------------------------------------------------------------------------


async def extract_facts(
    grandma_message: str,
    memory_context: dict,
    conversation_history: list[dict],
    existing_facts: list[str],
) -> list[dict]:
    """Ask Claude to extract NEW personal facts from *grandma_message*.

    Args:
        grandma_message:       The message we are analysing.
        memory_context:        Row from the memories table (used for context).
        conversation_history:  Session turns so far (list of role/content dicts).
        existing_facts:        Fact strings already stored for this grandma.

    Returns:
        List of dicts, each with "fact" (str) and "confidence" (float 0–1).
        Facts below _MIN_CONFIDENCE are already filtered out.
        Deduplication against existing_facts is applied after extraction.
    """
    memory_summary = (
        memory_context.get("ai_summary")
        or memory_context.get("original_caption")
        or "a cherished family photo"
    )

    system = """\
You are a careful, empathetic assistant that extracts personal facts about an elderly \
woman from her conversation messages. You output ONLY a valid JSON array — no prose, \
no markdown, no explanations.

Each element has exactly two keys:
  "fact"       — a simple, declarative sentence in third person that captures ONE \
specific, personal detail about her (e.g. "Margaret grew roses in her Pasadena garden")
  "confidence" — a float from 0.0 to 1.0 reflecting how certain you are:
                   1.0 = explicit, direct statement
                   0.7 = strong implication from context
                   0.4 = plausible inference (minimum included)
                   below 0.4 = omit the fact entirely

Rules:
- Extract ONLY new facts not already in the "Known facts" list.
- Facts must be about the person herself: her life, family, places, emotions, \
hobbies, history — not about the photo or general observations.
- Each fact must be a single, atomic statement (one idea per fact).
- If there are no new facts to extract, return an empty array: []
- Output NOTHING except the JSON array.\
"""

    user = (
        f"Photo context: {memory_summary}\n\n"
        f"Recent conversation:\n{_format_history(conversation_history)}\n\n"
        f"Grandma's latest message: {grandma_message}\n\n"
        f"Known facts (do NOT repeat these):\n{_format_existing_facts(existing_facts)}\n\n"
        "Extract new personal facts from Grandma's latest message. Return a JSON array."
    )

    logger.debug("[profile_extractor] Sending extraction request to Claude")

    try:
        raw = await _call_extraction(system, user)
    except Exception as exc:
        logger.warning("[profile_extractor] Claude extraction failed: %s", exc)
        return []

    # --- Parse JSON ----------------------------------------------------------
    try:
        # Claude sometimes wraps output in ```json ... ``` — strip it.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        facts: list[dict[str, Any]] = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("[profile_extractor] JSON parse error: %s | raw=%r", exc, raw)
        return []

    if not isinstance(facts, list):
        logger.warning("[profile_extractor] Unexpected response shape: %r", facts)
        return []

    # --- Filter & validate ---------------------------------------------------
    valid: list[dict] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        if not fact or confidence < _MIN_CONFIDENCE:
            continue
        valid.append({"fact": fact, "confidence": round(confidence, 2)})

    logger.debug("[profile_extractor] Extracted %d valid facts", len(valid))
    return deduplicate_facts(valid, existing_facts)


# ---------------------------------------------------------------------------
# Public: deduplicate_facts
# ---------------------------------------------------------------------------


def deduplicate_facts(
    new_facts: list[dict],
    existing_facts: list[str],
) -> list[dict]:
    """Remove facts from *new_facts* that overlap significantly with *existing_facts*.

    Overlap is measured by counting shared significant (non-stop) words.
    A new fact is dropped if it shares ≥ _OVERLAP_THRESHOLD words with any
    existing fact.

    Args:
        new_facts:      List of {"fact": str, "confidence": float} dicts.
        existing_facts: Plain fact strings already stored for this grandma.

    Returns:
        Filtered copy of *new_facts*.
    """
    existing_word_sets = [_significant_words(f) for f in existing_facts]
    result: list[dict] = []
    for item in new_facts:
        new_words = _significant_words(item["fact"])
        is_dup = any(
            len(new_words & existing) >= _OVERLAP_THRESHOLD
            for existing in existing_word_sets
        )
        if not is_dup:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Public: save_extracted_facts
# ---------------------------------------------------------------------------


async def save_extracted_facts(
    grandma_id: str,
    session_id: str,
    facts: list[dict],
) -> None:
    """Persist *facts* to grandma_profile_facts (fire-and-forget safe).

    Uses asyncio.to_thread so the synchronous supabase-py call does not block
    the event loop.

    Args:
        grandma_id: UUID of the grandma row.
        session_id: UUID of the current session (stored for provenance).
        facts:      List of {"fact": str, "confidence": float} dicts.
    """
    if not facts:
        return

    import asyncio

    from .database import add_profile_fact  # import here to avoid circular at module load

    async def _insert_all() -> None:
        for item in facts:
            try:
                await asyncio.to_thread(
                    add_profile_fact,
                    grandma_id,
                    item["fact"],
                    session_id,
                    item.get("confidence", 1.0),
                )
                logger.debug("[profile_extractor] Saved fact: %s", item["fact"])
            except Exception as exc:
                logger.warning("[profile_extractor] Failed to save fact %r: %s", item, exc)

    await _insert_all()


# ---------------------------------------------------------------------------
# Convenience: extract + save in one call (intended for asyncio.create_task)
# ---------------------------------------------------------------------------


async def extract_and_save(
    grandma_id: str,
    session_id: str,
    grandma_message: str,
    memory_context: dict,
    conversation_history: list[dict],
    existing_facts: list[str],
) -> None:
    """Extract facts from *grandma_message* and save them to the DB.

    Designed to be fire-and-forgotten via asyncio.create_task():

        asyncio.create_task(extract_and_save(
            grandma_id, session_id, message, memory, history, existing
        ))

    All errors are logged and swallowed so a failure here never affects
    the main response flow.
    """
    try:
        facts = await extract_facts(
            grandma_message, memory_context, conversation_history, existing_facts
        )
        await save_extracted_facts(grandma_id, session_id, facts)
    except Exception as exc:
        logger.error("[profile_extractor] Unhandled error in extract_and_save: %s", exc)
