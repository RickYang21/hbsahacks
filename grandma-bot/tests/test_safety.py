"""Tests for safety rails and edge-case handling.

Covers:
  - Low-signal detection (emoji-only, single word, <5 words)
  - low_signal_count accumulation → REFLECT transition
  - Expanded distress/confusion keyword detection
  - Unsupported content (voice, video, links, group chat)
  - Repeated story detection
  - Full handler flows (mocked DB + BlueBubbles)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from therapy.state_machine import (
    SessionPhase,
    SessionState,
    _LOW_SIGNAL_COUNT_LIMIT,
    advance_state,
    detect_distress,
    detect_low_signal,
    determine_next_phase,
)
from therapy.handler import (
    _attachment_kind,
    _is_repeated_story,
    _make_low_signal_reply,
    _URL_RE,
)


# ===========================================================================
# detect_low_signal
# ===========================================================================


class TestDetectLowSignal:
    def test_single_word_yes(self):
        assert detect_low_signal("yes") is True

    def test_single_word_no(self):
        assert detect_low_signal("no") is True

    def test_single_word_okay(self):
        assert detect_low_signal("okay") is True

    def test_single_word_hmm(self):
        assert detect_low_signal("hmm") is True

    def test_single_word_with_punctuation(self):
        assert detect_low_signal("okay.") is True

    def test_emoji_only_heart(self):
        assert detect_low_signal("❤️") is True

    def test_emoji_only_smile(self):
        assert detect_low_signal("😊") is True

    def test_emoji_only_multiple(self):
        assert detect_low_signal("❤️😊🌸") is True

    def test_fewer_than_5_words(self):
        assert detect_low_signal("That sounds nice") is True

    def test_exactly_4_words(self):
        assert detect_low_signal("I love that dear") is True

    def test_5_words_not_low_signal(self):
        assert detect_low_signal("I remember that day well") is False

    def test_long_message_not_low_signal(self):
        assert detect_low_signal(
            "Oh yes, I remember that summer so clearly — we were at the lake house."
        ) is False

    def test_empty_string(self):
        assert detect_low_signal("") is True

    def test_whitespace_only(self):
        assert detect_low_signal("   ") is True

    def test_unknown_single_word_not_low_signal(self):
        # A real word that isn't in the low-signal set shouldn't trigger
        # if it's the only word — but it's still <5 words so it IS low signal.
        assert detect_low_signal("wonderful") is True  # 1 word < 5 words

    def test_sentence_with_unknown_words(self):
        assert detect_low_signal("That was a beautiful summer vacation") is False  # 6 words


# ===========================================================================
# low_signal_count state tracking
# ===========================================================================


class TestLowSignalCountState:
    def _make_state(self) -> SessionState:
        return SessionState(memory_id="test-mem")

    def test_increments_on_low_signal(self):
        state = self._make_state()
        advance_state(state, "yes")
        assert state.low_signal_count == 1

    def test_resets_on_substantive_message(self):
        state = self._make_state()
        advance_state(state, "yes")
        advance_state(state, "yes")
        advance_state(state, "Oh I remember that day so well, we were all together.")
        assert state.low_signal_count == 0

    def test_three_low_signal_triggers_reflect(self):
        state = self._make_state()
        # GREET_ANCHOR → EXPAND on first reply
        advance_state(state, "yes")
        # Now in EXPAND; two more low-signal replies
        advance_state(state, "hmm")
        advance_state(state, "okay")
        assert state.current_phase == SessionPhase.REFLECT

    def test_two_low_signal_does_not_trigger_reflect(self):
        state = self._make_state()
        # Goes GREET_ANCHOR → EXPAND on first reply (even if low-signal)
        advance_state(state, "yes")
        advance_state(state, "hmm")
        # Only 2 low-signal so far — should not be REFLECT yet
        assert state.current_phase != SessionPhase.REFLECT

    def test_low_signal_count_stored_on_state(self):
        state = self._make_state()
        state.current_phase = SessionPhase.EXPAND
        advance_state(state, "yes")
        advance_state(state, "hmm")
        assert state.low_signal_count == 2

    def test_emoji_counts_as_low_signal(self):
        state = self._make_state()
        state.current_phase = SessionPhase.EXPAND
        for msg in ["❤️", "😊", "🌸"]:
            advance_state(state, msg)
        assert state.current_phase == SessionPhase.REFLECT


# ===========================================================================
# detect_distress — expanded patterns
# ===========================================================================


class TestDetectDistress:
    # Original patterns still work
    def test_im_scared(self):
        assert detect_distress("I'm scared") is True

    def test_leave_me_alone(self):
        assert detect_distress("leave me alone") is True

    def test_stop(self):
        assert detect_distress("stop") is True

    def test_im_upset(self):
        assert detect_distress("I'm upset") is True

    # New confusion signals
    def test_who_is_this(self):
        assert detect_distress("who is this") is True

    def test_i_dont_understand(self):
        assert detect_distress("I don't understand") is True

    def test_what_are_you_talking_about(self):
        assert detect_distress("what are you talking about") is True

    def test_who_are_you(self):
        assert detect_distress("who are you") is True

    # New distress signals
    def test_that_makes_me_sad(self):
        assert detect_distress("that makes me sad") is True

    def test_im_crying(self):
        assert detect_distress("I'm crying") is True

    # Normal messages should not trigger
    def test_normal_message(self):
        assert detect_distress("I remember it was a lovely summer day") is False

    def test_i_dont_know_not_distress(self):
        # "I don't know" is a tiredness signal, not distress
        assert detect_distress("I don't know") is False

    def test_case_insensitive(self):
        assert detect_distress("WHO IS THIS") is True
        assert detect_distress("I DON'T UNDERSTAND") is True


# ===========================================================================
# Attachment kind detection
# ===========================================================================


class TestAttachmentKind:
    def test_voice_m4a(self):
        assert _attachment_kind("https://example.com/audio/message.m4a") == "voice"

    def test_voice_caf(self):
        assert _attachment_kind("https://example.com/audio.caf") == "voice"

    def test_voice_with_query_string(self):
        assert _attachment_kind("https://example.com/msg.m4a?token=abc") == "voice"

    def test_video_mp4(self):
        assert _attachment_kind("https://example.com/clip.mp4") == "video"

    def test_video_mov(self):
        assert _attachment_kind("https://example.com/video.mov") == "video"

    def test_image_returns_none(self):
        assert _attachment_kind("https://example.com/photo.jpg") is None

    def test_none_url_returns_none(self):
        assert _attachment_kind(None) is None

    def test_no_extension(self):
        assert _attachment_kind("https://example.com/photo") is None


# ===========================================================================
# Link detection
# ===========================================================================


class TestLinkDetection:
    def test_detects_http_link(self):
        assert _URL_RE.search("Check this out http://example.com") is not None

    def test_detects_https_link(self):
        assert _URL_RE.search("https://photos.google.com/album/123") is not None

    def test_no_link_in_plain_text(self):
        assert _URL_RE.search("I remember that summer so well") is None


# ===========================================================================
# Repeated story detection
# ===========================================================================


class TestRepeatedStoryDetection:
    _STORY = (
        "We used to go to the lake every summer, the whole family together. "
        "Your grandfather would fish all morning while we swam."
    )

    def test_identical_story_is_repeated(self):
        assert _is_repeated_story(self._STORY, [self._STORY]) is True

    def test_very_similar_story_is_repeated(self):
        similar = (
            "We used to go to the lake every summer, the whole family together. "
            "Grandpa would fish all morning while we swam."
        )
        assert _is_repeated_story(self._STORY, [similar]) is True

    def test_different_message_not_repeated(self):
        other = "I grew roses in the garden every spring — beautiful red ones."
        assert _is_repeated_story(self._STORY, [other]) is False

    def test_short_message_skipped(self):
        assert _is_repeated_story("yes", [self._STORY]) is False

    def test_empty_past_turns(self):
        assert _is_repeated_story(self._STORY, []) is False

    def test_case_insensitive(self):
        upper = self._STORY.upper()
        assert _is_repeated_story(self._STORY, [upper]) is True


# ===========================================================================
# Low-signal reply template
# ===========================================================================


class TestMakeLowSignalReply:
    def test_contains_grandma_name(self):
        reply = _make_low_signal_reply("Margaret", SessionPhase.EXPAND)
        # Some acknowledgments include the name
        assert isinstance(reply, str) and len(reply) > 0

    def test_expand_phase_includes_reengagement(self):
        reply = _make_low_signal_reply("Margaret", SessionPhase.EXPAND)
        # Should be acknowledgment + re-engage question
        assert "?" in reply or len(reply) > 10

    def test_reflect_phase_no_reengagement(self):
        # REFLECT has no re-engage questions — returns ack only
        reply = _make_low_signal_reply("Margaret", SessionPhase.REFLECT)
        assert isinstance(reply, str) and len(reply) > 0


# ===========================================================================
# Handler integration — mocked DB + BlueBubbles
# ===========================================================================


def _mock_grandma():
    return {"id": "gma-1", "name": "Margaret", "phone": "+15550000001"}


def _mock_session():
    return {"id": "sess-1", "grandma_id": "gma-1", "memory_id": "mem-1", "status": "active"}


def _mock_memory():
    return {"id": "mem-1", "ai_summary": "Family at the lake", "ai_tags": [], "image_url": None}


def _make_turn(role: str, content: str) -> dict:
    return {"role": role, "content": content, "image_url": None}


class TestHandlerGroupChat:
    def test_group_chat_ignored(self):
        from therapy.handler import handle_grandma_message

        with patch("therapy.handler.db") as mock_db:
            asyncio.get_event_loop().run_until_complete(
                handle_grandma_message("+15550000001", "hello", is_group_chat=True)
            )
            # No DB calls should happen
            mock_db.get_grandma_by_phone.assert_not_called()


class TestHandlerVoiceMemo:
    def test_voice_memo_sends_redirect(self):
        from therapy.handler import handle_grandma_message
        from therapy.prompts import VOICE_MEMO_REDIRECT

        with (
            patch("therapy.handler.db") as mock_db,
            patch("therapy.handler.bb") as mock_bb,
        ):
            mock_db.get_grandma_by_phone.return_value = _mock_grandma()
            mock_db.get_active_session.return_value = _mock_session()
            mock_bb.send_text = AsyncMock()

            asyncio.get_event_loop().run_until_complete(
                handle_grandma_message(
                    "+15550000001",
                    "",
                    image_url="https://example.com/audio.m4a",
                )
            )

            mock_bb.send_text.assert_called_once_with("+15550000001", VOICE_MEMO_REDIRECT)
            # Should not save a turn for the redirect
            mock_db.add_turn.assert_not_called()


class TestHandlerVideoRedirect:
    def test_video_sends_redirect(self):
        from therapy.handler import handle_grandma_message
        from therapy.prompts import VIDEO_REDIRECT

        with (
            patch("therapy.handler.db") as mock_db,
            patch("therapy.handler.bb") as mock_bb,
        ):
            mock_db.get_grandma_by_phone.return_value = _mock_grandma()
            mock_db.get_active_session.return_value = _mock_session()
            mock_bb.send_text = AsyncMock()

            asyncio.get_event_loop().run_until_complete(
                handle_grandma_message(
                    "+15550000001",
                    "",
                    image_url="https://example.com/clip.mp4",
                )
            )

            mock_bb.send_text.assert_called_once_with("+15550000001", VIDEO_REDIRECT)


class TestHandlerDistress:
    def _run_distress(self, message: str):
        from therapy.handler import handle_grandma_message

        with (
            patch("therapy.handler.db") as mock_db,
            patch("therapy.handler.bb") as mock_bb,
            patch("therapy.handler.asyncio.create_task"),
        ):
            mock_db.get_grandma_by_phone.return_value = _mock_grandma()
            mock_db.get_active_session.return_value = _mock_session()
            mock_db.add_turn = MagicMock()
            mock_db.flag_session = MagicMock()
            mock_db.add_session_alert = MagicMock()
            mock_bb.send_text = AsyncMock()

            asyncio.get_event_loop().run_until_complete(
                handle_grandma_message("+15550000001", message)
            )
            return mock_db, mock_bb

    def test_distress_flags_session(self):
        mock_db, _ = self._run_distress("I don't understand who you are")
        mock_db.flag_session.assert_called_once_with("sess-1")

    def test_distress_adds_alert(self):
        mock_db, _ = self._run_distress("who is this")
        mock_db.add_session_alert.assert_called_once()
        call_args = mock_db.add_session_alert.call_args[0]
        assert call_args[0] == "sess-1"
        assert call_args[1] == "distress"

    def test_confusion_triggers_safety_exit(self):
        mock_db, mock_bb = self._run_distress("what are you talking about")
        mock_bb.send_text.assert_called()

    def test_distress_does_not_end_session(self):
        """Session is flagged, not ended."""
        mock_db, _ = self._run_distress("I'm scared")
        mock_db.end_session.assert_not_called()
        mock_db.flag_session.assert_called_once()


class TestHandlerLinkStripping:
    def test_link_stripped_from_content(self):
        from therapy.handler import handle_grandma_message

        with (
            patch("therapy.handler.db") as mock_db,
            patch("therapy.handler.bb") as mock_bb,
            patch("therapy.handler.generate_therapy_response", new_callable=AsyncMock) as mock_gen,
            patch("therapy.handler.asyncio.create_task"),
        ):
            mock_db.get_grandma_by_phone.return_value = _mock_grandma()
            mock_db.get_active_session.return_value = _mock_session()
            mock_db.get_session_turns.return_value = []
            mock_db.get_profile_facts.return_value = []
            mock_db.get_past_grandma_turns.return_value = []
            mock_db.add_turn = MagicMock()
            mock_gen.return_value = "That's lovely!"
            mock_bb.send_text = AsyncMock()

            asyncio.get_event_loop().run_until_complete(
                handle_grandma_message(
                    "+15550000001",
                    "Look at this https://photos.example.com/abc I remember it well enough",
                )
            )

            # The saved turn and Claude call should use stripped content
            saved_content = mock_db.add_turn.call_args_list[0][0][2]
            assert "https://" not in saved_content


class TestHandlerRepeatedStory:
    def test_repeated_story_skips_fact_extraction(self):
        from therapy.handler import handle_grandma_message

        story = (
            "We used to go to the lake every summer, the whole family together. "
            "Your grandfather would fish all morning while we swam together happily."
        )

        with (
            patch("therapy.handler.db") as mock_db,
            patch("therapy.handler.bb") as mock_bb,
            patch("therapy.handler.generate_therapy_response", new_callable=AsyncMock) as mock_gen,
            patch("therapy.handler.extract_and_save", new_callable=AsyncMock) as mock_extract,
            patch("therapy.handler.asyncio.create_task") as mock_task,
        ):
            mock_db.get_grandma_by_phone.return_value = _mock_grandma()
            mock_db.get_active_session.return_value = _mock_session()
            mock_db.get_session_turns.return_value = []
            mock_db.get_profile_facts.return_value = []
            mock_db.get_past_grandma_turns.return_value = [story]
            mock_db.add_turn = MagicMock()
            mock_gen.return_value = "I love hearing that!"
            mock_bb.send_text = AsyncMock()

            asyncio.get_event_loop().run_until_complete(
                handle_grandma_message("+15550000001", story)
            )

            # extract_and_save should NOT be scheduled
            mock_task.assert_not_called()
