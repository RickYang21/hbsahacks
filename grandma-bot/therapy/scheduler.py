"""Background task management for therapy session lifecycle.

Four public coroutines
----------------------
timeout_checker_loop()          — poll every 5 min, close stale sessions
can_start_session(grandma_id)   — cooldown gate (True/"" or False/reason)
schedule_daily_session(...)     — daily auto-session trigger
family_trigger_session(...)     — family-initiated session with status string

Lifespan helper
---------------
therapy_lifespan(app)  — asynccontextmanager that starts/stops the checker

Usage in main.py
----------------
    from contextlib import asynccontextmanager
    from therapy.scheduler import therapy_lifespan

    # Option A — dedicated lifespan (if main.py has no lifespan yet)
    app = FastAPI(lifespan=therapy_lifespan)

    # Option B — compose inside an existing lifespan
    @asynccontextmanager
    async def lifespan(app):
        async with therapy_lifespan(app):
            yield
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

from . import database as db
from .bluebubbles import client as bb
from .handler import select_next_memory, start_session
from .profile_extractor import extract_and_save

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIMEOUT_POLL_SECONDS = 5 * 60          # check every 5 minutes
_TIMEOUT_MINUTES = 30                   # sessions idle this long get closed
_COOLDOWN_HOURS = 4                     # minimum gap between sessions
_TIMEOUT_CLOSE_MESSAGE = (
    "It was so lovely talking with you, Mom. "
    "Let's chat again soon! 💛"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_all_active_sessions() -> list[dict]:
    """Return all sessions with status='active', joined with grandma + turns."""
    r = (
        db._db()
        .table("sessions")
        .select("*, grandmas(*), memories(*)")
        .eq("status", "active")
        .execute()
    )
    return r.data or []


def _minutes_since_last_turn(session_id: str) -> float | None:
    """Return minutes elapsed since the most recent turn, or None if no turns."""
    r = (
        db._db()
        .table("turns")
        .select("created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not r.data:
        return None
    raw = r.data[0].get("created_at")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds() / 60
    except (ValueError, AttributeError):
        return None


async def _close_timed_out_session(session: dict) -> None:
    """Send a warm close message, run fact extraction, and end the session."""
    session_id: str = session["id"]
    grandma: dict = session.get("grandmas") or {}
    grandma_id: str = grandma.get("id") or session.get("grandma_id", "")
    grandma_name: str = grandma.get("name") or "there"
    grandma_phone: str = grandma.get("phone") or ""
    memory: dict = session.get("memories") or {}

    logger.info(
        "[scheduler] Closing timed-out session %s for grandma %s",
        session_id,
        grandma_name,
    )

    # Send the gentle goodbye
    try:
        await bb.send_text(grandma_phone, _TIMEOUT_CLOSE_MESSAGE)
    except Exception as exc:
        logger.error("[scheduler] send_text for timeout close failed: %s", exc)

    # Save the bot's goodbye turn
    try:
        await asyncio.to_thread(
            db.add_turn, session_id, "bot", _TIMEOUT_CLOSE_MESSAGE
        )
    except Exception as exc:
        logger.error("[scheduler] add_turn for timeout close failed: %s", exc)

    # Run profile extraction on the full transcript before closing
    try:
        turns = await asyncio.to_thread(db.get_session_turns, session_id)
        profile_facts = await asyncio.to_thread(db.get_profile_facts, grandma_id)
        existing_fact_strings = [f["fact"] for f in profile_facts]

        # Extract from every grandma turn we haven't processed yet.
        # Concatenate all grandma messages as one extraction context.
        grandma_turns = [t for t in turns if t.get("role") == "grandma"]
        for turn in grandma_turns:
            content = (turn.get("content") or "").strip()
            if content:
                await extract_and_save(
                    grandma_id,
                    session_id,
                    content,
                    memory,
                    turns,
                    existing_fact_strings,
                )
    except Exception as exc:
        logger.error("[scheduler] fact extraction on timeout failed: %s", exc)

    # End the session
    try:
        await asyncio.to_thread(db.end_session, session_id)
    except Exception as exc:
        logger.error("[scheduler] end_session on timeout failed: %s", exc)


# ---------------------------------------------------------------------------
# timeout_checker_loop
# ---------------------------------------------------------------------------


async def timeout_checker_loop() -> None:
    """Runs forever as a background task — polls for stale sessions every 5 min.

    For each active session idle for ≥ _TIMEOUT_MINUTES it sends a warm
    goodbye, runs profile extraction on the transcript, and ends the session.
    """
    logger.info("[scheduler] Timeout checker started (poll=%ds)", _TIMEOUT_POLL_SECONDS)
    while True:
        try:
            await _check_timeouts()
        except Exception as exc:
            # Never crash the loop — just log and keep going.
            logger.error("[scheduler] Unexpected error in timeout loop: %s", exc)
        await asyncio.sleep(_TIMEOUT_POLL_SECONDS)


async def _check_timeouts() -> None:
    try:
        sessions = await asyncio.to_thread(_get_all_active_sessions)
    except Exception as exc:
        logger.error("[scheduler] _get_all_active_sessions failed: %s", exc)
        return

    if not sessions:
        return

    for session in sessions:
        session_id = session.get("id", "?")
        try:
            idle_minutes = await asyncio.to_thread(
                _minutes_since_last_turn, session_id
            )
            if idle_minutes is None:
                # No turns yet — check time since session started
                started_raw = session.get("started_at")
                if started_raw:
                    try:
                        started = datetime.fromisoformat(
                            str(started_raw).replace("Z", "+00:00")
                        )
                        if started.tzinfo is None:
                            started = started.replace(tzinfo=timezone.utc)
                        idle_minutes = (
                            datetime.now(timezone.utc) - started
                        ).total_seconds() / 60
                    except (ValueError, AttributeError):
                        idle_minutes = 0.0
                else:
                    idle_minutes = 0.0

            if idle_minutes >= _TIMEOUT_MINUTES:
                logger.info(
                    "[scheduler] Session %s idle %.0f min — closing",
                    session_id,
                    idle_minutes,
                )
                await _close_timed_out_session(session)
        except Exception as exc:
            logger.error(
                "[scheduler] Error processing session %s: %s", session_id, exc
            )


# ---------------------------------------------------------------------------
# can_start_session
# ---------------------------------------------------------------------------


async def can_start_session(grandma_id: str) -> tuple[bool, str]:
    """Cooldown gate — check whether a new session is allowed right now.

    Returns:
        (True, "")                            — session may be started
        (False, "human-readable reason")      — cooldown not yet expired
    """
    try:
        last = await asyncio.to_thread(db.get_last_ended_session, grandma_id)
    except Exception as exc:
        logger.warning("[scheduler] can_start_session DB error: %s", exc)
        return True, ""  # fail open so a DB hiccup doesn't block the session

    if last is None:
        return True, ""

    ended_raw = last.get("ended_at") or last.get("started_at")
    if not ended_raw:
        return True, ""

    try:
        ended_at = datetime.fromisoformat(str(ended_raw).replace("Z", "+00:00"))
        if ended_at.tzinfo is None:
            ended_at = ended_at.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return True, ""

    from datetime import timedelta

    elapsed = datetime.now(timezone.utc) - ended_at
    cooldown = timedelta(hours=_COOLDOWN_HOURS)

    if elapsed < cooldown:
        remaining = cooldown - elapsed
        hours, remainder = divmod(int(remaining.total_seconds()), 3600)
        minutes = remainder // 60
        elapsed_h = int(elapsed.total_seconds() // 3600)
        reason = (
            f"Last session ended {elapsed_h} hour(s) ago; "
            f"cooldown is {_COOLDOWN_HOURS} hours. "
            f"Ready in {hours}h {minutes}m."
        )
        return False, reason

    return True, ""


# ---------------------------------------------------------------------------
# schedule_daily_session
# ---------------------------------------------------------------------------


async def schedule_daily_session(
    grandma_id: str,
    preferred_hour: int = 10,
) -> None:
    """Daily background task — trigger a session at *preferred_hour* (local time).

    For the hackathon this is called once and loops forever using asyncio.sleep.
    In production you'd replace this with APScheduler or a cron job.

    Args:
        grandma_id:     UUID of the grandma to chat with.
        preferred_hour: Hour of day (0–23) to attempt a session (server local time).
    """
    logger.info(
        "[scheduler] Daily session scheduler started for grandma %s at hour %d",
        grandma_id,
        preferred_hour,
    )
    while True:
        now = datetime.now()
        # Sleep until preferred_hour today (or tomorrow if already past)
        target = now.replace(hour=preferred_hour, minute=0, second=0, microsecond=0)
        if target <= now:
            from datetime import timedelta
            target = target + timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        logger.debug(
            "[scheduler] Next daily session attempt in %.0f seconds", wait_seconds
        )
        await asyncio.sleep(wait_seconds)

        try:
            # Check cooldown
            ok, reason = await can_start_session(grandma_id)
            if not ok:
                logger.info("[scheduler] Daily session skipped — %s", reason)
                continue

            # Pick a memory
            memory_id = await select_next_memory(grandma_id)
            if memory_id is None:
                logger.info(
                    "[scheduler] Daily session skipped — no unused memories for grandma %s",
                    grandma_id,
                )
                continue

            logger.info(
                "[scheduler] Triggering daily session: grandma=%s memory=%s",
                grandma_id,
                memory_id,
            )
            await start_session(grandma_id, memory_id)
        except Exception as exc:
            logger.error("[scheduler] Daily session trigger failed: %s", exc)


# ---------------------------------------------------------------------------
# family_trigger_session
# ---------------------------------------------------------------------------


async def family_trigger_session(grandma_id: str, memory_id: str) -> str:
    """Start a session from a family member's request.

    Called by Person A's handler when a family member texts a trigger phrase
    (e.g. "send mom the beach photo today").

    Args:
        grandma_id: UUID of the grandma.
        memory_id:  UUID of the memory the family member selected.

    Returns:
        A short status string suitable for sending back to the family member.
    """
    # Check cooldown
    ok, reason = await can_start_session(grandma_id)
    if not ok:
        logger.info("[scheduler] family_trigger_session blocked — %s", reason)
        return f"Not quite yet! {reason}"

    # Validate memory exists
    try:
        memory = await asyncio.to_thread(db.get_memory, memory_id)
    except Exception as exc:
        logger.error("[scheduler] family_trigger_session: get_memory failed: %s", exc)
        return "Something went wrong looking up that memory — try again?"

    if memory is None:
        return "I couldn't find that memory. Can you try again?"

    # Guard: active session already running
    try:
        existing = await asyncio.to_thread(db.get_active_session, grandma_id)
    except Exception as exc:
        logger.error("[scheduler] family_trigger_session: active session check failed: %s", exc)
        return "Something went wrong — try again in a moment?"

    if existing is not None:
        return "Mom is already in a session right now — check back soon! 💛"

    # Kick off session as background task so we can reply to family immediately
    summary = memory.get("ai_summary") or memory.get("original_caption") or "that photo"
    asyncio.create_task(start_session(grandma_id, memory_id))

    logger.info(
        "[scheduler] family_trigger_session: starting session grandma=%s memory=%s",
        grandma_id,
        memory_id,
    )
    return f"Starting a session with Mom using {summary}! 💛"


# ---------------------------------------------------------------------------
# Lifespan context manager (for FastAPI)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def therapy_lifespan(app: "FastAPI"):
    """FastAPI lifespan that starts the timeout checker on startup.

    Usage — dedicated lifespan (main.py has no existing lifespan):
        app = FastAPI(lifespan=therapy_lifespan)

    Usage — composing with an existing lifespan:
        @asynccontextmanager
        async def lifespan(app):
            async with therapy_lifespan(app):
                # ...existing startup logic...
                yield
                # ...existing shutdown logic...
    """
    task = asyncio.create_task(timeout_checker_loop())
    logger.info("[scheduler] Timeout checker task created")
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        logger.info("[scheduler] Timeout checker task cancelled")
