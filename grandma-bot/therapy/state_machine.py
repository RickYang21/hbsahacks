"""Conversation state machine for reminiscence therapy sessions.

Tracks which phase of a session we're in and drives phase transitions
purely from turn counts and lightweight keyword matching.  No LLM calls
are made here — this runs in the hot path of every inbound message.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------


class SessionPhase(str, Enum):
    """Ordered phases of a reminiscence therapy session."""

    GREET_ANCHOR = "greet_anchor"  # 1 — photo sent, waiting for first reply
    EXPAND = "expand"              # 2 — who / what / when / where
    DEEPEN = "deepen"              # 3 — feelings, meaning, favourites
    REFLECT = "reflect"            # 4 — summary, warmth, goodbye
    ENDED = "ended"                # 5 — session closed normally
    SAFETY_EXIT = "safety_exit"    # ∞ — distress detected, flag & redirect


# ---------------------------------------------------------------------------
# Thresholds (override per-session if desired)
# ---------------------------------------------------------------------------


@dataclass
class PhaseThresholds:
    """Configurable turn limits.  *_min is the earliest a phase can end;
    *_max is the latest before we force-advance."""

    expand_min: int = 1    # bot exchanges before EXPAND can end
    expand_max: int = 2    # bot exchanges before EXPAND must end
    deepen_min: int = 3    # spend more time on deep questions
    deepen_max: int = 5
    max_total_turns: int = 14      # hard cap across entire session
    timeout_minutes: int = 30      # minutes without a reply → auto-close


DEFAULT_THRESHOLDS = PhaseThresholds()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """All mutable state for one therapy session."""

    memory_id: str
    current_phase: SessionPhase = SessionPhase.GREET_ANCHOR
    turn_count: int = 0           # total turns (bot + grandma combined)
    phase_turn_count: int = 0     # bot turns *in the current phase*
    last_message_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    thresholds: PhaseThresholds = field(default_factory=PhaseThresholds)
    # How many consecutive short replies grandma has given (fatigue signal).
    consecutive_short_replies: int = 0
    # How many consecutive low-signal replies (emoji/single-word/<5 words).
    low_signal_count: int = 0


# ---------------------------------------------------------------------------
# Keyword detection helpers
# ---------------------------------------------------------------------------

# Patterns are compiled once at import time.

_TIREDNESS_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bI'?m\s+tired\b",
    r"\bso\s+tired\b",
    r"\bmaybe\s+later\b",
    r"\bnot\s+now\b",
    r"\bI\s+don'?t\s+know\b",
    r"\bI\s+can'?t\s+remember\b",
    r"\bI\s+don'?t\s+remember\b",
    r"\bI\s+forget\b",
    r"\bI'?m\s+not\s+sure\b",
    r"\bI\s+need\s+to\s+rest\b",
    r"\bI'?m\s+getting\s+tired\b",
    r"\blet'?s\s+talk\s+another\s+time\b",
    r"\btalk\s+later\b",
    r"\bI'?m\s+done\b",
    r"\benough\s+for\s+today\b",
    r"\bI'?m\s+sleepy\b",
]]

_DISTRESS_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    # Distress signals
    r"\bI'?m\s+scared\b",
    r"\bI'?m\s+confused\b",
    r"\bI'?m\s+upset\b",
    r"\bI'?m\s+frightened\b",
    r"\bthat\s+makes\s+me\s+sad\b",
    r"\bI'?m\s+crying\b",
    r"\bI\s+feel\s+sad\b",
    r"\bI'?m\s+not\s+okay\b",
    r"\bsomething'?s?\s+wrong\b",
    r"\bhelp\s+me\b",
    r"\bI\s+want\s+to\s+go\s+home\b",
    r"\bI'?m\s+lost\b",
    r"\bwhere\s+am\s+I\b",
    r"\bwhat'?s?\s+happening\b",
    # Confusion signals
    r"\bwho\s+is\s+this\b",
    r"\bwho\s+are\s+you\b",
    r"\bI\s+don'?t\s+know\s+you\b",
    r"\bI\s+don'?t\s+understand\b",
    r"\bwhat\s+are\s+you\s+talking\s+about\b",
    r"\bI\s+don'?t\s+know\s+what\s+you\s+mean\b",
    # Rejection signals
    r"\bstop\b",
    r"\bleave\s+me\s+alone\b",
    r"\bgo\s+away\b",
]]

# A reply this short (chars, stripped) counts as a "short reply".
_SHORT_REPLY_THRESHOLD = 15
# Two consecutive short replies trigger fatigue-based early exit.
_CONSECUTIVE_SHORT_LIMIT = 2

# Low-signal detection: single common words that carry little conversational content.
_LOW_SIGNAL_WORDS: frozenset[str] = frozenset({
    "yes", "no", "okay", "ok", "hmm", "hm", "sure", "maybe", "alright",
    "yep", "nope", "yeah", "nah", "fine", "good", "nice", "oh", "ah",
    "uh", "um", "huh", "right", "true", "great",
})
# Three consecutive low-signal replies → transition to REFLECT.
_LOW_SIGNAL_COUNT_LIMIT = 3

# Matches strings that contain only emoji, variation selectors, and whitespace.
_EMOJI_STRIP_RE = re.compile(
    r"[\U0001F300-\U0001FAFF"   # Misc Symbols, Emoticons, Transport & Map, etc.
    r"\U00002600-\U000027BF"    # Misc Symbols & Dingbats
    r"\U0000FE00-\U0000FE0F"    # Variation Selectors
    r"\u200d"                   # Zero Width Joiner
    r"\uFE0F"                   # Variation Selector-16
    r"\s]+",
    re.UNICODE,
)


def detect_tiredness(message: str) -> bool:
    """Return True if *message* contains fatigue or avoidance signals."""
    return any(p.search(message) for p in _TIREDNESS_PATTERNS)


def detect_distress(message: str) -> bool:
    """Return True if *message* contains confusion or distress signals."""
    return any(p.search(message) for p in _DISTRESS_PATTERNS)


def _is_short_reply(message: str) -> bool:
    return len(message.strip()) <= _SHORT_REPLY_THRESHOLD


def _is_emoji_only(message: str) -> bool:
    """Return True if *message* contains only emoji, variation selectors, and whitespace."""
    text = message.strip()
    if not text:
        return False
    return _EMOJI_STRIP_RE.fullmatch(text) is not None


def detect_low_signal(message: str) -> bool:
    """Return True if *message* is low-signal: emoji-only, single common word, or <5 words."""
    text = message.strip()
    if not text:
        return True
    if _is_emoji_only(text):
        return True
    words = text.lower().split()
    if len(words) == 1 and words[0].rstrip(".,!?") in _LOW_SIGNAL_WORDS:
        return True
    if len(words) < 5:
        return True
    return False


# ---------------------------------------------------------------------------
# Force-close check
# ---------------------------------------------------------------------------


def should_force_close(state: SessionState) -> tuple[bool, str]:
    """Check whether the session must be closed regardless of phase.

    Returns (True, reason_string) when closing is required, (False, "") otherwise.
    """
    if state.current_phase in (SessionPhase.ENDED, SessionPhase.SAFETY_EXIT):
        return False, ""

    now = datetime.now(timezone.utc)
    elapsed_minutes = (now - state.last_message_at).total_seconds() / 60
    if elapsed_minutes >= state.thresholds.timeout_minutes:
        return True, f"timeout:{elapsed_minutes:.0f}min"

    if state.turn_count >= state.thresholds.max_total_turns:
        return True, f"max_turns:{state.turn_count}"

    return False, ""


# ---------------------------------------------------------------------------
# Phase transition logic
# ---------------------------------------------------------------------------


def determine_next_phase(
    state: SessionState,
    grandma_message: str,
) -> SessionPhase:
    """Return the phase the session should move into after grandma's message.

    This function is *pure* — it does not mutate *state*.  The caller is
    responsible for applying the returned phase and incrementing counters.

    Transition rules (checked in priority order):
      1. Distress detected → SAFETY_EXIT
      2. Tiredness detected → REFLECT (skip ahead)
      3. Consecutive short replies threshold hit → REFLECT
      4. Normal phase progression based on phase_turn_count
      5. Stay in current phase if thresholds not yet met
    """
    current = state.current_phase

    # Terminal phases never advance further.
    if current in (SessionPhase.ENDED, SessionPhase.SAFETY_EXIT):
        return current

    # ── Priority 1: safety ──────────────────────────────────────────────────
    if detect_distress(grandma_message):
        return SessionPhase.SAFETY_EXIT

    # ── Priority 2 & 3: fatigue signals ────────────────────────────────────
    consecutive = state.consecutive_short_replies + (1 if _is_short_reply(grandma_message) else 0)

    if detect_tiredness(grandma_message) or consecutive >= _CONSECUTIVE_SHORT_LIMIT:
        # Only skip forward if there's somewhere meaningful to skip to.
        if current not in (SessionPhase.REFLECT, SessionPhase.ENDED):
            return SessionPhase.REFLECT

    # ── Priority 3b: low-signal streak ─────────────────────────────────────
    # low_signal_count is already updated in advance_state before this call.
    if state.low_signal_count >= _LOW_SIGNAL_COUNT_LIMIT:
        if current not in (SessionPhase.REFLECT, SessionPhase.ENDED):
            return SessionPhase.REFLECT

    # ── Priority 4: normal progression ─────────────────────────────────────
    t = state.thresholds

    if current == SessionPhase.GREET_ANCHOR:
        # Any reply from grandma moves us into EXPAND.
        return SessionPhase.EXPAND

    if current == SessionPhase.EXPAND:
        # phase_turn_count counts *bot* turns in this phase.
        # We advance when the bot has asked enough questions.
        if state.phase_turn_count >= t.expand_min:
            return SessionPhase.DEEPEN
        return SessionPhase.EXPAND

    if current == SessionPhase.DEEPEN:
        if state.phase_turn_count >= t.deepen_min:
            return SessionPhase.REFLECT
        return SessionPhase.DEEPEN

    if current == SessionPhase.REFLECT:
        # Bot sends the reflection message, then we mark ENDED.
        # This transition happens when the bot has sent at least one
        # reflection turn — the caller signals that by incrementing
        # phase_turn_count before calling us again (or by checking
        # phase_turn_count >= 1 after the bot's reflection reply).
        if state.phase_turn_count >= 1:
            return SessionPhase.ENDED
        return SessionPhase.REFLECT

    return current  # fallback — should not be reached


# ---------------------------------------------------------------------------
# Convenience: apply a grandma-message event to state in-place
# ---------------------------------------------------------------------------


def advance_state(state: SessionState, grandma_message: str) -> SessionPhase:
    """Update *state* in-place after receiving *grandma_message* and return
    the new phase.

    Increments turn_count, updates last_message_at, tracks short-reply
    streak, then delegates to determine_next_phase.
    """
    state.turn_count += 1
    state.last_message_at = datetime.now(timezone.utc)

    if _is_short_reply(grandma_message):
        state.consecutive_short_replies += 1
    else:
        state.consecutive_short_replies = 0

    if detect_low_signal(grandma_message):
        state.low_signal_count += 1
    else:
        state.low_signal_count = 0

    new_phase = determine_next_phase(state, grandma_message)

    if new_phase != state.current_phase:
        state.current_phase = new_phase
        state.phase_turn_count = 0  # reset counter on phase change
    # Note: phase_turn_count for *bot* turns is incremented externally
    # (after the bot replies) to keep this function focused on grandma events.

    return new_phase


def record_bot_turn(state: SessionState) -> None:
    """Call this after the bot sends its reply to advance phase_turn_count.

    Also checks whether the session should advance to ENDED after a
    REFLECT reply (the bot's closing message triggers the final transition).
    """
    state.turn_count += 1
    state.phase_turn_count += 1

    # After REFLECT the bot has sent the goodbye — close the session.
    if state.current_phase == SessionPhase.REFLECT and state.phase_turn_count >= 1:
        state.current_phase = SessionPhase.ENDED
        state.phase_turn_count = 0

    # Force advance in EXPAND / DEEPEN if bot has hit the hard max.
    t = state.thresholds
    if state.current_phase == SessionPhase.EXPAND and state.phase_turn_count >= t.expand_max:
        state.current_phase = SessionPhase.DEEPEN
        state.phase_turn_count = 0
    elif state.current_phase == SessionPhase.DEEPEN and state.phase_turn_count >= t.deepen_max:
        state.current_phase = SessionPhase.REFLECT
        state.phase_turn_count = 0
