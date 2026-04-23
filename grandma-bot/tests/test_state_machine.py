"""Unit tests for therapy/state_machine.py.

Covers every transition path, both early-exit conditions, the timeout /
max-turns force-close check, and the keyword detectors.
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta, timezone

import pytest

# Make sure the project root is on the path so the therapy package is importable
# even when tests are run from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from therapy.state_machine import (
    DEFAULT_THRESHOLDS,
    PhaseThresholds,
    SessionPhase,
    SessionState,
    advance_state,
    detect_distress,
    detect_tiredness,
    determine_next_phase,
    record_bot_turn,
    should_force_close,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(**kwargs) -> SessionState:
    """Create a SessionState with sensible defaults, overridable via kwargs."""
    defaults = dict(
        memory_id="mem-001",
        current_phase=SessionPhase.GREET_ANCHOR,
        turn_count=0,
        phase_turn_count=0,
        thresholds=PhaseThresholds(),
    )
    defaults.update(kwargs)
    return SessionState(**defaults)


# ---------------------------------------------------------------------------
# detect_tiredness
# ---------------------------------------------------------------------------


class TestDetectTiredness:
    def test_im_tired(self):
        assert detect_tiredness("I'm tired, dear")

    def test_maybe_later(self):
        assert detect_tiredness("Maybe later, okay?")

    def test_not_now(self):
        assert detect_tiredness("Not now, sweetheart")

    def test_i_dont_know(self):
        assert detect_tiredness("I don't know about that")

    def test_i_cant_remember(self):
        assert detect_tiredness("I can't remember anymore")

    def test_enough_for_today(self):
        assert detect_tiredness("Enough for today!")

    def test_i_need_to_rest(self):
        assert detect_tiredness("I need to rest now")

    def test_talk_later(self):
        assert detect_tiredness("Let's talk later")

    def test_normal_message_is_not_tired(self):
        assert not detect_tiredness("Oh yes, that was a beautiful summer!")

    def test_case_insensitive(self):
        assert detect_tiredness("I'M TIRED")

    def test_i_forget(self):
        assert detect_tiredness("I forget what year that was")


# ---------------------------------------------------------------------------
# detect_distress
# ---------------------------------------------------------------------------


class TestDetectDistress:
    def test_im_scared(self):
        assert detect_distress("I'm scared")

    def test_im_confused(self):
        assert detect_distress("I'm confused")

    def test_who_are_you(self):
        assert detect_distress("Who are you?")

    def test_stop(self):
        assert detect_distress("Stop, please stop")

    def test_leave_me_alone(self):
        assert detect_distress("Leave me alone!")

    def test_help_me(self):
        assert detect_distress("Help me please")

    def test_where_am_i(self):
        assert detect_distress("Where am I?")

    def test_i_want_to_go_home(self):
        assert detect_distress("I want to go home")

    def test_normal_message_is_not_distress(self):
        assert not detect_distress("That was the best summer of my life.")

    def test_case_insensitive(self):
        assert detect_distress("WHO ARE YOU")


# ---------------------------------------------------------------------------
# Normal phase transitions (determine_next_phase)
# ---------------------------------------------------------------------------


class TestNormalTransitions:
    def test_greet_anchor_any_reply_goes_to_expand(self):
        state = make_state(current_phase=SessionPhase.GREET_ANCHOR)
        assert determine_next_phase(state, "Oh my, that takes me back!") == SessionPhase.EXPAND

    def test_expand_stays_until_min_bot_turns(self):
        # phase_turn_count = 1, min = 2 → stay in EXPAND
        state = make_state(
            current_phase=SessionPhase.EXPAND,
            phase_turn_count=1,
            thresholds=PhaseThresholds(expand_min=2),
        )
        assert determine_next_phase(state, "It was a lovely day.") == SessionPhase.EXPAND

    def test_expand_advances_at_min_bot_turns(self):
        state = make_state(
            current_phase=SessionPhase.EXPAND,
            phase_turn_count=2,
            thresholds=PhaseThresholds(expand_min=2),
        )
        assert determine_next_phase(state, "We had a picnic.") == SessionPhase.DEEPEN

    def test_deepen_stays_until_min_bot_turns(self):
        state = make_state(
            current_phase=SessionPhase.DEEPEN,
            phase_turn_count=1,
            thresholds=PhaseThresholds(deepen_min=2),
        )
        assert determine_next_phase(state, "It felt wonderful.") == SessionPhase.DEEPEN

    def test_deepen_advances_at_min_bot_turns(self):
        state = make_state(
            current_phase=SessionPhase.DEEPEN,
            phase_turn_count=2,
            thresholds=PhaseThresholds(deepen_min=2),
        )
        assert determine_next_phase(state, "That was my favourite day.") == SessionPhase.REFLECT

    def test_reflect_advances_to_ended_after_one_bot_turn(self):
        state = make_state(
            current_phase=SessionPhase.REFLECT,
            phase_turn_count=1,  # bot has sent the goodbye
        )
        assert determine_next_phase(state, "Thank you dear.") == SessionPhase.ENDED

    def test_reflect_stays_when_bot_hasnt_replied_yet(self):
        state = make_state(current_phase=SessionPhase.REFLECT, phase_turn_count=0)
        assert determine_next_phase(state, "That was nice.") == SessionPhase.REFLECT

    def test_ended_stays_ended(self):
        state = make_state(current_phase=SessionPhase.ENDED)
        assert determine_next_phase(state, "Hello?") == SessionPhase.ENDED

    def test_safety_exit_stays(self):
        state = make_state(current_phase=SessionPhase.SAFETY_EXIT)
        assert determine_next_phase(state, "I'm lost") == SessionPhase.SAFETY_EXIT


# ---------------------------------------------------------------------------
# Early exit transitions
# ---------------------------------------------------------------------------


class TestEarlyExits:
    def test_distress_from_greet_anchor(self):
        state = make_state(current_phase=SessionPhase.GREET_ANCHOR)
        assert determine_next_phase(state, "Who are you? I'm scared!") == SessionPhase.SAFETY_EXIT

    def test_distress_from_expand(self):
        state = make_state(current_phase=SessionPhase.EXPAND, phase_turn_count=1)
        assert determine_next_phase(state, "I'm confused and scared") == SessionPhase.SAFETY_EXIT

    def test_distress_from_deepen(self):
        state = make_state(current_phase=SessionPhase.DEEPEN, phase_turn_count=1)
        assert determine_next_phase(state, "Leave me alone!") == SessionPhase.SAFETY_EXIT

    def test_tiredness_skips_to_reflect(self):
        state = make_state(current_phase=SessionPhase.EXPAND, phase_turn_count=1)
        assert determine_next_phase(state, "I'm tired, maybe later") == SessionPhase.REFLECT

    def test_tiredness_from_deepen_skips_to_reflect(self):
        state = make_state(current_phase=SessionPhase.DEEPEN, phase_turn_count=1)
        assert determine_next_phase(state, "I don't remember anymore") == SessionPhase.REFLECT

    def test_distress_takes_priority_over_tiredness(self):
        # message contains both — distress wins
        state = make_state(current_phase=SessionPhase.EXPAND, phase_turn_count=1)
        result = determine_next_phase(state, "I'm tired and confused, who are you?")
        assert result == SessionPhase.SAFETY_EXIT

    def test_consecutive_short_replies_trigger_reflect(self):
        # 1 prior short reply already on state, second short reply → reflect
        state = make_state(
            current_phase=SessionPhase.EXPAND,
            phase_turn_count=1,
            consecutive_short_replies=1,
        )
        assert determine_next_phase(state, "Yes.") == SessionPhase.REFLECT

    def test_single_short_reply_does_not_trigger_early_exit(self):
        state = make_state(
            current_phase=SessionPhase.EXPAND,
            phase_turn_count=1,
            consecutive_short_replies=0,
        )
        assert determine_next_phase(state, "Yes.") == SessionPhase.EXPAND

    def test_long_reply_resets_short_count_via_advance_state(self):
        state = make_state(
            current_phase=SessionPhase.EXPAND,
            phase_turn_count=1,
            consecutive_short_replies=1,
        )
        advance_state(state, "Oh what a wonderful memory, that brings me so much joy!")
        assert state.consecutive_short_replies == 0


# ---------------------------------------------------------------------------
# should_force_close
# ---------------------------------------------------------------------------


class TestShouldForceClose:
    def test_no_close_normally(self):
        state = make_state(current_phase=SessionPhase.EXPAND, turn_count=3)
        ok, reason = should_force_close(state)
        assert not ok
        assert reason == ""

    def test_max_turns_triggers_close(self):
        state = make_state(
            current_phase=SessionPhase.EXPAND,
            turn_count=10,
            thresholds=PhaseThresholds(max_total_turns=10),
        )
        ok, reason = should_force_close(state)
        assert ok
        assert "max_turns" in reason

    def test_timeout_triggers_close(self):
        old_time = datetime.now(timezone.utc) - timedelta(minutes=31)
        state = make_state(
            current_phase=SessionPhase.DEEPEN,
            thresholds=PhaseThresholds(timeout_minutes=30),
        )
        state.last_message_at = old_time
        ok, reason = should_force_close(state)
        assert ok
        assert "timeout" in reason

    def test_no_close_when_already_ended(self):
        state = make_state(current_phase=SessionPhase.ENDED, turn_count=15)
        ok, _ = should_force_close(state)
        assert not ok

    def test_no_close_when_safety_exit(self):
        state = make_state(current_phase=SessionPhase.SAFETY_EXIT, turn_count=15)
        ok, _ = should_force_close(state)
        assert not ok

    def test_just_under_timeout_does_not_close(self):
        almost_old = datetime.now(timezone.utc) - timedelta(minutes=29)
        state = make_state(
            current_phase=SessionPhase.EXPAND,
            thresholds=PhaseThresholds(timeout_minutes=30),
        )
        state.last_message_at = almost_old
        ok, _ = should_force_close(state)
        assert not ok


# ---------------------------------------------------------------------------
# advance_state + record_bot_turn integration
# ---------------------------------------------------------------------------


class TestAdvanceStateIntegration:
    def test_full_happy_path(self):
        """Walk through all phases with minimal turns.

        Phase advances are grandma-message-driven: determine_next_phase()
        checks phase_turn_count (bot turns in the current phase) each time
        grandma replies, and advances the phase when the threshold is met.
        record_bot_turn() increments phase_turn_count and enforces hard maxes.
        """
        state = make_state(thresholds=PhaseThresholds(expand_min=2, deepen_min=2))

        # GREET_ANCHOR → EXPAND on first grandma reply
        phase = advance_state(state, "Oh, that looks lovely!")
        assert phase == SessionPhase.EXPAND
        assert state.phase_turn_count == 0

        # Bot turn 1 in EXPAND (phase_turn_count → 1, still < expand_min=2)
        record_bot_turn(state)
        assert state.phase_turn_count == 1
        assert state.current_phase == SessionPhase.EXPAND

        # Grandma reply: phase_turn_count=1 < expand_min=2 → stay in EXPAND
        advance_state(state, "We were at the park.")
        assert state.current_phase == SessionPhase.EXPAND

        # Bot turn 2 in EXPAND (phase_turn_count → 2, hits expand_min)
        record_bot_turn(state)
        assert state.phase_turn_count == 2
        assert state.current_phase == SessionPhase.EXPAND  # bot turn alone doesn't advance

        # Grandma reply: phase_turn_count=2 >= expand_min=2 → advance to DEEPEN
        advance_state(state, "It was a warm afternoon.")
        assert state.current_phase == SessionPhase.DEEPEN
        assert state.phase_turn_count == 0

        # Bot turn 1 in DEEPEN (phase_turn_count → 1, still < deepen_min=2)
        record_bot_turn(state)
        assert state.phase_turn_count == 1
        assert state.current_phase == SessionPhase.DEEPEN

        # Grandma reply: phase_turn_count=1 < deepen_min=2 → stay in DEEPEN
        advance_state(state, "I felt so happy.")
        assert state.current_phase == SessionPhase.DEEPEN

        # Bot turn 2 in DEEPEN (phase_turn_count → 2, hits deepen_min)
        record_bot_turn(state)
        assert state.phase_turn_count == 2
        assert state.current_phase == SessionPhase.DEEPEN  # bot turn alone doesn't advance

        # Grandma reply: phase_turn_count=2 >= deepen_min=2 → advance to REFLECT
        advance_state(state, "It was my mother's birthday.")
        assert state.current_phase == SessionPhase.REFLECT
        assert state.phase_turn_count == 0

        # Bot sends the reflection/goodbye message → immediately marks ENDED
        record_bot_turn(state)
        assert state.current_phase == SessionPhase.ENDED

    def test_early_exit_to_safety(self):
        state = make_state(current_phase=SessionPhase.EXPAND, phase_turn_count=1)
        advance_state(state, "Who are you? Leave me alone!")
        assert state.current_phase == SessionPhase.SAFETY_EXIT

    def test_early_exit_to_reflect_on_tiredness(self):
        state = make_state(current_phase=SessionPhase.DEEPEN, phase_turn_count=1)
        advance_state(state, "I'm tired, not now dear")
        assert state.current_phase == SessionPhase.REFLECT

    def test_turn_count_increments_on_both_sides(self):
        state = make_state()
        advance_state(state, "Hello!")      # grandma turn
        record_bot_turn(state)             # bot turn
        assert state.turn_count == 2

    def test_expand_hard_max_via_record_bot_turn(self):
        """Bot exceeding expand_max forces advance to DEEPEN even without grandma input."""
        state = make_state(
            current_phase=SessionPhase.EXPAND,
            phase_turn_count=0,
            thresholds=PhaseThresholds(expand_max=2),
        )
        record_bot_turn(state)  # phase_turn_count → 1
        assert state.current_phase == SessionPhase.EXPAND
        record_bot_turn(state)  # phase_turn_count → 2 == expand_max → advance
        assert state.current_phase == SessionPhase.DEEPEN

    def test_deepen_hard_max_via_record_bot_turn(self):
        state = make_state(
            current_phase=SessionPhase.DEEPEN,
            phase_turn_count=0,
            thresholds=PhaseThresholds(deepen_max=2),
        )
        record_bot_turn(state)
        assert state.current_phase == SessionPhase.DEEPEN
        record_bot_turn(state)
        assert state.current_phase == SessionPhase.REFLECT

    def test_custom_thresholds(self):
        """Expand with expand_min=1 should move to DEEPEN after a single bot turn."""
        state = make_state(
            current_phase=SessionPhase.EXPAND,
            phase_turn_count=0,
            thresholds=PhaseThresholds(expand_min=1, expand_max=1),
        )
        record_bot_turn(state)
        assert state.current_phase == SessionPhase.DEEPEN
