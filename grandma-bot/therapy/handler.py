"""Full therapy session orchestrator — wires all sub-modules together.

Person A calls two entry points:
  handle_grandma_message(phone, content, image_url) — on every inbound message
  start_session(grandma_id, memory_id)              — to kick off a new session

All DB calls are wrapped in asyncio.to_thread() (supabase-py is synchronous).
Every step is wrapped in try/except — errors are logged and a graceful fallback
is sent so grandma never sees an error message.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import PurePosixPath
from urllib.parse import urlparse

from . import database as db
from .bluebubbles import client as bb
from .conversation import (
    generate_opener,
    generate_safety_response,
    generate_therapy_response,
)
from .profile_extractor import extract_and_save
from .prompts import (
    LOW_SIGNAL_ACKNOWLEDGMENTS,
    LOW_SIGNAL_REENGAGES,
    VOICE_MEMO_REDIRECT,
    VIDEO_REDIRECT,
)
from .state_machine import (
    PhaseThresholds,
    SessionPhase,
    SessionState,
    advance_state,
    detect_distress,
    detect_low_signal,
    record_bot_turn,
    should_force_close,
)

logger = logging.getLogger(__name__)

_FALLBACK_REPLY = "I'll talk to you soon, {grandma_name} 💛"
_SESSION_COOLDOWN_HOURS = 4

# Per-phone lock — serializes concurrent webhook handling for the same grandma.
_grandma_locks: dict[str, asyncio.Lock] = {}

# Cooldown for the "no active session" holding message so it's only sent once per hour.
_holding_sent: dict[str, float] = {}
_HOLDING_COOLDOWN = 3600.0

# ---------------------------------------------------------------------------
# Attachment / content-type helpers
# ---------------------------------------------------------------------------

_VOICE_EXTENSIONS = frozenset({".m4a", ".caf", ".aiff", ".aif", ".mp3", ".wav", ".amr", ".ogg"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".m4v", ".mkv", ".3gp", ".webm"})
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# Similarity ratio above which a message is considered a repeated story.
_REPEATED_STORY_THRESHOLD = 0.65
# Minimum message length (chars) before we bother checking for repeats.
_REPEATED_STORY_MIN_LEN = 25


def _attachment_kind(image_url: str | None) -> str | None:
    """Return 'voice', 'video', or None based on the file extension in *image_url*."""
    if not image_url:
        return None
    try:
        path = PurePosixPath(urlparse(image_url).path)
        ext = path.suffix.lower()
    except Exception:
        return None
    if ext in _VOICE_EXTENSIONS:
        return "voice"
    if ext in _VIDEO_EXTENSIONS:
        return "video"
    return None


def _is_repeated_story(message: str, past_turns: list[str]) -> bool:
    """Return True if *message* is closely similar to any previous grandma message.

    Uses SequenceMatcher ratio — fast enough for ≤60 past turns.
    Only compares messages that are long enough to carry a story.
    """
    if len(message.strip()) < _REPEATED_STORY_MIN_LEN or not past_turns:
        return False
    msg_lower = message.lower()
    for past in past_turns:
        if not past or len(past) < _REPEATED_STORY_MIN_LEN:
            continue
        ratio = SequenceMatcher(None, msg_lower, past.lower()).ratio()
        if ratio >= _REPEATED_STORY_THRESHOLD:
            return True
    return False


def _make_low_signal_reply(grandma_name: str, phase: SessionPhase) -> str:
    """Build a warm acknowledgment + gentle re-engagement message."""
    ack = random.choice(LOW_SIGNAL_ACKNOWLEDGMENTS).format(grandma_name=grandma_name)
    reengages = LOW_SIGNAL_REENGAGES.get(phase, [])
    if not reengages:
        return ack
    return f"{ack} {random.choice(reengages)}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reconstruct_state(session: dict, turns: list[dict]) -> SessionState:
    """Replay all recorded turns to rebuild an in-memory SessionState.

    This is always correct regardless of what is stored in the DB, and
    requires no extra schema columns.  With ≤10 turns per session the
    overhead is negligible.
    """
    state = SessionState(memory_id=session.get("memory_id") or "")
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content") or ""
        if role == "grandma":
            advance_state(state, content)
        elif role == "bot":
            record_bot_turn(state)
    return state


def _update_phase_in_db(session_id: str, phase: SessionPhase) -> None:
    """Best-effort: persist current_phase to the sessions row for API reads.

    If the column doesn't exist in the schema this silently fails — it is
    a convenience for dashboard reads, not load-bearing.
    """
    try:
        db._db().table("sessions").update(
            {"current_phase": phase.value}
        ).eq("id", session_id).execute()
    except Exception as exc:
        logger.debug("[handler] _update_phase_in_db skipped (column may not exist): %s", exc)


def _cooldown_ok(grandma_id: str) -> bool:
    """Return True if ≥ _SESSION_COOLDOWN_HOURS have passed since the last session."""
    last = db.get_last_ended_session(grandma_id)
    if last is None:
        return True
    ended_raw = last.get("ended_at") or last.get("started_at")
    if not ended_raw:
        return True
    try:
        ended_at = datetime.fromisoformat(str(ended_raw).replace("Z", "+00:00"))
        if ended_at.tzinfo is None:
            ended_at = ended_at.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ended_at) >= timedelta(hours=_SESSION_COOLDOWN_HOURS)
    except (ValueError, AttributeError):
        return True


async def _send_fallback(phone: str, grandma_name: str = "there") -> None:
    try:
        await bb.send_text(phone, _FALLBACK_REPLY.format(grandma_name=grandma_name))
    except Exception as exc:
        logger.error("[handler] fallback send failed: %s", exc)


async def _notify_family(grandma_id: str, grandma_name: str) -> None:
    """Send a gentle heads-up text to all registered family members."""
    try:
        family = await asyncio.to_thread(db.get_family_members, grandma_id)
    except Exception as exc:
        logger.error("[handler] get_family_members failed: %s", exc)
        return

    message = (
        f"Heads up — {grandma_name} seemed a bit confused during our chat today. "
        "She's okay, we wrapped up gently. 💛"
    )
    for member in family:
        phone = member.get("phone")
        if not phone:
            continue
        try:
            await bb.send_text(phone, message)
            logger.info("[handler] Notified family member %s", phone)
        except Exception as exc:
            logger.error("[handler] family notification to %s failed: %s", phone, exc)


# ---------------------------------------------------------------------------
# handle_grandma_message
# ---------------------------------------------------------------------------


async def handle_grandma_message(
    phone: str,
    content: str,
    image_url: str | None = None,
    is_group_chat: bool = False,
) -> None:
    """Entry point: called by Person A's webhook router for every grandma message.

    Args:
        phone:          Grandma's iMessage phone number (E.164 format).
        content:        The text body of her message.
        image_url:      Public URL of an attachment, if any.
        is_group_chat:  True when the message arrived in a group thread — ignored.
    """
    # ── 0. Ignore group-chat messages ─────────────────────────────────────
    if is_group_chat:
        logger.debug("[handler] Ignoring group-chat message from %s", phone)
        return

    # Serialize concurrent webhook events for the same phone so we never
    # send two replies in parallel for back-to-back messages.
    lock = _grandma_locks.setdefault(phone, asyncio.Lock())
    async with lock:
        await _handle_grandma_message_locked(phone, content, image_url)


async def _handle_grandma_message_locked(
    phone: str,
    content: str,
    image_url: str | None = None,
) -> None:
    # ── a. Look up grandma ────────────────────────────────────────────────
    try:
        grandma = await asyncio.to_thread(db.get_grandma_by_phone, phone)
    except Exception as exc:
        logger.error("[handler] DB lookup failed for phone=%s: %s", phone, exc)
        return

    if grandma is None:
        logger.warning("[handler] Unknown phone %s — ignoring", phone)
        return

    grandma_id: str = grandma["id"]
    grandma_name: str = grandma.get("name") or "there"
    grandma_phone: str = grandma.get("phone") or phone

    # ── b. Get active session ─────────────────────────────────────────────
    try:
        session = await asyncio.to_thread(db.get_active_session, grandma_id)
    except Exception as exc:
        logger.error("[handler] get_active_session failed: %s", exc)
        return

    # ── c. No active session → holding message (once per cooldown period) ──
    if session is None:
        logger.info("[handler] No active session for grandma_id=%s", grandma_id)
        now = time.time()
        if now - _holding_sent.get(grandma_phone, 0) > _HOLDING_COOLDOWN:
            _holding_sent[grandma_phone] = now
            try:
                await bb.send_text(
                    grandma_phone,
                    f"Hi {grandma_name}! I'll have something to share with you soon 💛",
                )
            except Exception as exc:
                logger.error("[handler] send holding message failed: %s", exc)
        else:
            logger.debug("[handler] Skipping repeated holding message for %s", grandma_phone)
        return

    session_id: str = session["id"]

    # ── c2. Unsupported content checks ───────────────────────────────────
    attachment = _attachment_kind(image_url)
    if attachment == "voice":
        logger.info("[handler] Voice memo from %s — sending redirect", phone)
        try:
            await bb.send_text(grandma_phone, VOICE_MEMO_REDIRECT.format(grandma_name=grandma_name))
        except Exception as exc:
            logger.error("[handler] voice redirect send failed: %s", exc)
        return
    if attachment == "video":
        logger.info("[handler] Video from %s — sending redirect", phone)
        try:
            await bb.send_text(grandma_phone, VIDEO_REDIRECT)
        except Exception as exc:
            logger.error("[handler] video redirect send failed: %s", exc)
        return

    # Links in the message body: strip URLs and continue normally.
    # We do NOT skip the message — grandma may have typed context around the link.
    clean_content = _URL_RE.sub("", content).strip() or content

    # ── d. Save inbound grandma turn ──────────────────────────────────────
    try:
        await asyncio.to_thread(db.add_turn, session_id, "grandma", clean_content, image_url)
    except Exception as exc:
        # Non-fatal — still try to respond
        logger.error("[handler] add_turn (grandma) failed: %s", exc)

    # ── e. Immediate distress/confusion check ─────────────────────────────
    if detect_distress(clean_content):
        logger.info("[handler] Distress detected — safety exit for session %s", session_id)
        safety_reply = generate_safety_response(grandma_name)
        try:
            await bb.send_text(grandma_phone, safety_reply)
            await asyncio.to_thread(db.add_turn, session_id, "bot", safety_reply)
            await asyncio.to_thread(db.flag_session, session_id)
        except Exception as exc:
            logger.error("[handler] Safety exit handling failed: %s", exc)
        try:
            await asyncio.to_thread(
                db.add_session_alert, session_id, "distress", clean_content
            )
        except Exception as exc:
            logger.error("[handler] add_session_alert failed: %s", exc)
        asyncio.create_task(_notify_family(grandma_id, grandma_name))
        return

    # ── f. Load turns + profile facts + past turns, reconstruct state ────
    try:
        turns, profile_facts, past_grandma_turns = await asyncio.gather(
            asyncio.to_thread(db.get_session_turns, session_id),
            asyncio.to_thread(db.get_profile_facts, grandma_id),
            asyncio.to_thread(db.get_past_grandma_turns, grandma_id),
        )
    except Exception as exc:
        logger.error("[handler] DB reads failed: %s", exc)
        await _send_fallback(grandma_phone, grandma_name)
        return

    state = _reconstruct_state(session, turns)
    existing_fact_strings = [f["fact"] for f in profile_facts]

    # ── g. Force-close check (timeout / max turns) ────────────────────────
    force_close, reason = should_force_close(state)
    if force_close:
        logger.info("[handler] Force-closing session %s — reason: %s", session_id, reason)
        # Drop into REFLECT so we generate a warm goodbye
        state.current_phase = SessionPhase.REFLECT

    # Already ENDED (shouldn't normally happen mid-message, but handle it)
    if state.current_phase == SessionPhase.ENDED:
        logger.info("[handler] Session %s already ENDED — closing DB record", session_id)
        await asyncio.to_thread(db.end_session, session_id)
        return

    # ── g2. Low-signal shortcut ───────────────────────────────────────────
    # If grandma sent a short/ambiguous reply AND we're not already winding
    # down, reply with a warm acknowledgment + a simpler re-engage question
    # instead of passing the low-signal message to Claude.
    # (If low_signal_count has already hit the limit, state is now REFLECT
    #  and we fall through to the normal Claude goodbye flow.)
    if (
        detect_low_signal(clean_content)
        and state.current_phase not in (SessionPhase.REFLECT, SessionPhase.ENDED, SessionPhase.SAFETY_EXIT)
    ):
        logger.info(
            "[handler] Low-signal reply (count=%d) — using acknowledgment template",
            state.low_signal_count,
        )
        low_signal_reply = _make_low_signal_reply(grandma_name, state.current_phase)
        try:
            await bb.send_text(grandma_phone, low_signal_reply)
            await asyncio.to_thread(db.add_turn, session_id, "bot", low_signal_reply)
        except Exception as exc:
            logger.error("[handler] Low-signal reply send/save failed: %s", exc)
        record_bot_turn(state)
        await asyncio.to_thread(_update_phase_in_db, session_id, state.current_phase)
        if state.current_phase in (SessionPhase.ENDED, SessionPhase.SAFETY_EXIT):
            await asyncio.to_thread(db.end_session, session_id)
        return

    # ── g3. Repeated-story detection ─────────────────────────────────────
    # Check BEFORE fact extraction so we don't save stale facts.
    is_repeated = _is_repeated_story(clean_content, past_grandma_turns)
    if is_repeated:
        logger.info("[handler] Repeated story detected — will skip fact extraction")

    # ── h. Resolve memory (joined into session or fetch separately) ───────
    memory: dict = {}
    if isinstance(session.get("memories"), dict):
        memory = session["memories"]
    else:
        try:
            fetched = await asyncio.to_thread(db.get_memory, session.get("memory_id") or "")
            memory = fetched or {}
        except Exception as exc:
            logger.warning("[handler] get_memory failed — continuing with empty: %s", exc)

    # ── i. Generate Claude response ───────────────────────────────────────
    try:
        reply = await generate_therapy_response(
            memory=memory,
            session_turns=turns,
            profile_facts=profile_facts,
            grandma_name=grandma_name,
            current_phase=state.current_phase,
            last_grandma_message=clean_content,
        )
    except Exception as exc:
        logger.error("[handler] generate_therapy_response failed: %s", exc)
        reply = _FALLBACK_REPLY.format(grandma_name=grandma_name)

    # ── j. Send reply via BlueBubbles ─────────────────────────────────────
    try:
        await bb.send_text(grandma_phone, reply)
    except Exception as exc:
        logger.error("[handler] send_text (reply) failed: %s", exc)

    # ── k. Save bot turn ──────────────────────────────────────────────────
    try:
        await asyncio.to_thread(db.add_turn, session_id, "bot", reply)
    except Exception as exc:
        logger.error("[handler] add_turn (bot) failed: %s", exc)

    # Advance phase_turn_count; may flip REFLECT → ENDED
    record_bot_turn(state)

    # ── l. Fire-and-forget profile fact extraction ────────────────────────
    # Skip if grandma is re-telling a story she's shared before — we don't
    # want to store duplicate facts, and interrupting her telling would be rude.
    if not is_repeated:
        asyncio.create_task(
            extract_and_save(
                grandma_id, session_id, clean_content, memory, turns, existing_fact_strings
            )
        )
    else:
        logger.info("[handler] Skipping fact extraction for repeated story")

    # ── m. Persist phase; close session if terminal ───────────────────────
    try:
        await asyncio.to_thread(_update_phase_in_db, session_id, state.current_phase)
        if state.current_phase in (SessionPhase.ENDED, SessionPhase.SAFETY_EXIT):
            await asyncio.to_thread(db.end_session, session_id)
            logger.info("[handler] Session %s closed (phase=%s)", session_id, state.current_phase.value)
    except Exception as exc:
        logger.error("[handler] phase/session update failed: %s", exc)


# ---------------------------------------------------------------------------
# start_session
# ---------------------------------------------------------------------------


async def start_session(grandma_id: str, memory_id: str) -> None:
    """Start a new therapy session: send photo + opener to grandma.

    Called by the dashboard "Start Session" button or the scheduler.

    Args:
        grandma_id: UUID of the grandma row.
        memory_id:  UUID of the memory to use for this session.
    """
    # ── a. Check for existing active session ──────────────────────────────
    try:
        existing = await asyncio.to_thread(db.get_active_session, grandma_id)
    except Exception as exc:
        logger.error("[handler] start_session: DB check failed: %s", exc)
        return

    if existing is not None:
        logger.warning(
            "[handler] start_session: session %s already active for grandma %s — aborting",
            existing["id"],
            grandma_id,
        )
        return

    # ── b. Cooldown check ─────────────────────────────────────────────────
    try:
        ok = await asyncio.to_thread(_cooldown_ok, grandma_id)
    except Exception as exc:
        logger.warning("[handler] start_session: cooldown check failed, proceeding: %s", exc)
        ok = True

    if not ok:
        logger.info(
            "[handler] start_session: cooldown not expired for grandma %s — skipping",
            grandma_id,
        )
        return

    # ── c + d. Fetch memory and grandma in parallel ───────────────────────
    try:
        memory, grandma = await asyncio.gather(
            asyncio.to_thread(db.get_memory, memory_id),
            asyncio.to_thread(db.get_grandma_by_id, grandma_id),
        )
    except Exception as exc:
        logger.error("[handler] start_session: data fetch failed: %s", exc)
        return

    if memory is None:
        logger.error("[handler] start_session: memory %s not found", memory_id)
        return
    if grandma is None:
        logger.error("[handler] start_session: grandma %s not found", grandma_id)
        return

    grandma_name: str = grandma.get("name") or "there"
    grandma_phone: str = grandma.get("phone") or ""

    # ── e. Fetch existing profile facts ───────────────────────────────────
    try:
        profile_facts = await asyncio.to_thread(db.get_profile_facts, grandma_id)
    except Exception as exc:
        logger.warning("[handler] start_session: profile facts fetch failed: %s", exc)
        profile_facts = []

    # ── f. Create session in DB ───────────────────────────────────────────
    try:
        session = await asyncio.to_thread(db.create_session, grandma_id, memory_id)
    except Exception as exc:
        logger.error("[handler] start_session: create_session failed: %s", exc)
        return

    session_id: str = session["id"]

    # ── g. Generate opener ────────────────────────────────────────────────
    try:
        opener = await generate_opener(memory, grandma_name, profile_facts)
    except Exception as exc:
        logger.error("[handler] start_session: generate_opener failed: %s", exc)
        opener = f"Hi {grandma_name}! I found something special to share with you today 💛"

    # ── h. Send photo ─────────────────────────────────────────────────────
    if memory.get("image_url"):
        try:
            await bb.send_image(grandma_phone, memory["image_url"])
        except Exception as exc:
            logger.error("[handler] start_session: send_image failed: %s", exc)

    # ── i. Small pause then send opener text ─────────────────────────────
    await asyncio.sleep(2)
    try:
        await bb.send_text(grandma_phone, opener)
    except Exception as exc:
        logger.error("[handler] start_session: send opener failed: %s", exc)

    # ── j. Save opener as first bot turn ──────────────────────────────────
    try:
        await asyncio.to_thread(
            db.add_turn, session_id, "bot", opener, memory.get("image_url")
        )
    except Exception as exc:
        logger.error("[handler] start_session: add_turn failed: %s", exc)

    # ── k. Mark memory as used ────────────────────────────────────────────
    try:
        await asyncio.to_thread(db.mark_memory_used, memory_id, session_id)
    except Exception as exc:
        logger.error("[handler] start_session: mark_memory_used failed: %s", exc)

    logger.info(
        "[handler] start_session complete: session_id=%s grandma=%s memory=%s",
        session_id,
        grandma_name,
        memory_id,
    )


# ---------------------------------------------------------------------------
# select_next_memory
# ---------------------------------------------------------------------------


async def select_next_memory(grandma_id: str) -> str | None:
    """Pick the best unused memory for the next session.

    Selection strategy (in priority order):
      1. Must not appear in used_in_sessions.
      2. Prefer a different era than the last session's memory (+2 pts).
      3. Prefer no tag overlap with the last session's memory (+1 pt).
      4. Fallback: most recently submitted (DB order, created_at DESC).

    Returns:
        memory_id string, or None if all memories have been used.
    """
    try:
        unused = await asyncio.to_thread(db.get_unused_memories, grandma_id)
    except Exception as exc:
        logger.error("[handler] select_next_memory failed: %s", exc)
        return None

    if not unused:
        logger.info("[handler] select_next_memory: no unused memories for grandma %s", grandma_id)
        return None

    # Gather last session's memory metadata for variety scoring
    last_era: str | None = None
    last_tags: set[str] = set()
    try:
        last_session = await asyncio.to_thread(db.get_last_ended_session, grandma_id)
        if last_session:
            last_mem = last_session.get("memories") or {}
            if isinstance(last_mem, dict):
                last_era = last_mem.get("era")
                last_tags = set(last_mem.get("ai_tags") or [])
    except Exception as exc:
        logger.debug("[handler] select_next_memory: last session fetch failed: %s", exc)

    def _score(m: dict) -> int:
        score = 0
        if last_era and m.get("era") and m.get("era") != last_era:
            score += 2
        m_tags = set(m.get("ai_tags") or [])
        if m_tags and not (m_tags & last_tags):
            score += 1
        return score

    if last_era or last_tags:
        best = max(unused, key=_score)
        if _score(best) > 0:
            logger.debug(
                "[handler] select_next_memory: chose %s (score=%d)", best["id"], _score(best)
            )
            return best["id"]

    # Default: most recently submitted (already ordered by DB query)
    return unused[0]["id"]
