"""Unit tests for therapy/profile_extractor.py.

Covers:
  - deduplicate_facts: overlap detection + stop-word stripping
  - extract_facts:     JSON parsing, confidence filtering, dedup integration
  - save_extracted_facts: calls add_profile_fact for each fact
  - extract_and_save: orchestration, swallows errors

Claude is mocked throughout; no real API calls are made.
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from therapy.profile_extractor import (
    _MIN_CONFIDENCE,
    _OVERLAP_THRESHOLD,
    _significant_words,
    deduplicate_facts,
    extract_and_save,
    extract_facts,
    save_extracted_facts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_memory(summary: str = "A birthday party in a garden") -> dict:
    return {"ai_summary": summary, "ai_tags": ["garden", "birthday"]}


def make_history() -> list[dict]:
    return [
        {"role": "bot", "content": "Hi Eleanor! Do you remember this photo?"},
        {"role": "grandma", "content": "Oh yes, that was my garden in Pasadena!"},
    ]


def claude_response(facts: list[dict]) -> MagicMock:
    """Build a fake anthropic response object returning *facts* as JSON."""
    block = MagicMock()
    block.text = json.dumps(facts)
    msg = MagicMock()
    msg.content = [block]
    return msg


# ---------------------------------------------------------------------------
# _significant_words
# ---------------------------------------------------------------------------


class TestSignificantWords:
    def test_strips_stop_words(self):
        words = _significant_words("I grew up in the garden")
        assert "i" not in words
        assert "the" not in words
        assert "in" not in words
        assert "grew" in words
        assert "garden" in words

    def test_strips_punctuation(self):
        words = _significant_words("roses, garden!")
        assert "roses" in words
        assert "garden" in words

    def test_empty_string(self):
        assert _significant_words("") == set()

    def test_all_stop_words(self):
        assert _significant_words("the a an") == set()


# ---------------------------------------------------------------------------
# deduplicate_facts
# ---------------------------------------------------------------------------


class TestDeduplicateFacts:
    def test_no_existing_facts_keeps_everything(self):
        new = [
            {"fact": "Eleanor grew roses in Pasadena", "confidence": 0.9},
            {"fact": "She loved baking apple pies", "confidence": 0.8},
        ]
        assert deduplicate_facts(new, []) == new

    def test_exact_overlap_removes_fact(self):
        existing = ["Eleanor grew roses in Pasadena garden"]
        new = [{"fact": "Eleanor grew roses Pasadena garden California", "confidence": 0.9}]
        # "eleanor", "grew", "roses", "pasadena", "garden" → 5 shared words → dup
        result = deduplicate_facts(new, existing)
        assert result == []

    def test_partial_overlap_below_threshold_keeps_fact(self):
        existing = ["Eleanor has a daughter named Susan"]
        # Only 1 shared significant word ("eleanor") — below OVERLAP_THRESHOLD
        new = [{"fact": "Eleanor visited Paris in 1962", "confidence": 0.9}]
        result = deduplicate_facts(new, existing)
        assert len(result) == 1

    def test_keeps_unrelated_fact(self):
        existing = ["Eleanor grew roses in her Pasadena garden"]
        new = [{"fact": "She loved baking apple pies on Sundays", "confidence": 0.8}]
        result = deduplicate_facts(new, existing)
        assert len(result) == 1

    def test_multiple_new_facts_partial_dedup(self):
        existing = ["Eleanor grew roses in Pasadena garden"]
        new = [
            {"fact": "Eleanor grew roses Pasadena California garden", "confidence": 0.9},  # dup
            {"fact": "She baked pies every Sunday morning", "confidence": 0.8},            # keep
        ]
        result = deduplicate_facts(new, existing)
        assert len(result) == 1
        assert "pies" in result[0]["fact"]

    def test_empty_new_facts(self):
        assert deduplicate_facts([], ["existing fact"]) == []

    def test_overlap_threshold_is_exactly_respected(self):
        # Build exactly OVERLAP_THRESHOLD shared significant words
        shared = ["apple", "orange", "banana"][:_OVERLAP_THRESHOLD]
        existing_fact = "She loved " + " ".join(shared) + " fruit"
        new_fact = "Eleanor enjoyed " + " ".join(shared) + " desserts"
        new = [{"fact": new_fact, "confidence": 0.9}]
        result = deduplicate_facts(new, [existing_fact])
        assert result == []  # exactly at threshold → removed


# ---------------------------------------------------------------------------
# extract_facts — mocking _call_extraction
# ---------------------------------------------------------------------------


class TestExtractFacts:
    @pytest.mark.asyncio
    async def test_returns_valid_facts(self):
        payload = [
            {"fact": "Eleanor grew roses in Pasadena", "confidence": 0.9},
            {"fact": "She had a daughter named Susan", "confidence": 0.7},
        ]
        with patch(
            "therapy.profile_extractor._call_extraction",
            new=AsyncMock(return_value=json.dumps(payload)),
        ):
            result = await extract_facts(
                "I grew up tending roses in my Pasadena garden.",
                make_memory(),
                make_history(),
                existing_facts=[],
            )
        assert len(result) == 2
        assert result[0]["fact"] == "Eleanor grew roses in Pasadena"
        assert result[0]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_filters_low_confidence(self):
        payload = [
            {"fact": "She might have liked swimming", "confidence": 0.3},   # below threshold
            {"fact": "Eleanor had a dog named Max", "confidence": 0.8},
        ]
        with patch(
            "therapy.profile_extractor._call_extraction",
            new=AsyncMock(return_value=json.dumps(payload)),
        ):
            result = await extract_facts(
                "I remember Max...",
                make_memory(),
                make_history(),
                existing_facts=[],
            )
        assert len(result) == 1
        assert "Max" in result[0]["fact"]

    @pytest.mark.asyncio
    async def test_exactly_at_min_confidence_kept(self):
        payload = [{"fact": "Eleanor enjoyed gardening", "confidence": _MIN_CONFIDENCE}]
        with patch(
            "therapy.profile_extractor._call_extraction",
            new=AsyncMock(return_value=json.dumps(payload)),
        ):
            result = await extract_facts("I like gardening.", make_memory(), [], [])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_deduplicates_against_existing(self):
        existing = ["Eleanor grew roses in her Pasadena garden"]
        payload = [
            # will be removed by dedup (shares Eleanor, grew, roses, Pasadena, garden)
            {"fact": "Eleanor grew roses Pasadena California garden", "confidence": 0.9},
            # kept
            {"fact": "She had a brother named Robert", "confidence": 0.8},
        ]
        with patch(
            "therapy.profile_extractor._call_extraction",
            new=AsyncMock(return_value=json.dumps(payload)),
        ):
            result = await extract_facts(
                "My brother Robert visited too.",
                make_memory(),
                make_history(),
                existing_facts=existing,
            )
        assert len(result) == 1
        assert "Robert" in result[0]["fact"]

    @pytest.mark.asyncio
    async def test_empty_array_response(self):
        with patch(
            "therapy.profile_extractor._call_extraction",
            new=AsyncMock(return_value="[]"),
        ):
            result = await extract_facts("Yes, lovely.", make_memory(), [], [])
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_json_wrapped_in_code_fence(self):
        payload = [{"fact": "Eleanor worked as a nurse", "confidence": 0.9}]
        fenced = f"```json\n{json.dumps(payload)}\n```"
        with patch(
            "therapy.profile_extractor._call_extraction",
            new=AsyncMock(return_value=fenced),
        ):
            result = await extract_facts(
                "I was a nurse for 30 years.", make_memory(), [], []
            )
        assert len(result) == 1
        assert "nurse" in result[0]["fact"]

    @pytest.mark.asyncio
    async def test_returns_empty_on_invalid_json(self):
        with patch(
            "therapy.profile_extractor._call_extraction",
            new=AsyncMock(return_value="not json at all"),
        ):
            result = await extract_facts("Hmm.", make_memory(), [], [])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_claude_error(self):
        with patch(
            "therapy.profile_extractor._call_extraction",
            new=AsyncMock(side_effect=Exception("connection error")),
        ):
            result = await extract_facts("Hello.", make_memory(), [], [])
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_non_dict_items(self):
        payload = ["bad item", {"fact": "Eleanor loved cats", "confidence": 0.8}]
        with patch(
            "therapy.profile_extractor._call_extraction",
            new=AsyncMock(return_value=json.dumps(payload)),
        ):
            result = await extract_facts("Cats are lovely.", make_memory(), [], [])
        assert len(result) == 1
        assert "cats" in result[0]["fact"].lower()


# ---------------------------------------------------------------------------
# save_extracted_facts
# ---------------------------------------------------------------------------


class TestSaveExtractedFacts:
    # add_profile_fact is imported lazily inside save_extracted_facts to avoid
    # circular imports, so we patch the canonical location: therapy.database.
    @pytest.mark.asyncio
    async def test_calls_add_profile_fact_for_each(self):
        facts = [
            {"fact": "Eleanor grew roses", "confidence": 0.9},
            {"fact": "She loved baking", "confidence": 0.7},
        ]
        mock_add = MagicMock()
        with patch("therapy.database.add_profile_fact", mock_add):
            await save_extracted_facts("gma-1", "sess-1", facts)
        assert mock_add.call_count == 2
        mock_add.assert_any_call("gma-1", "Eleanor grew roses", "sess-1", 0.9)
        mock_add.assert_any_call("gma-1", "She loved baking", "sess-1", 0.7)

    @pytest.mark.asyncio
    async def test_empty_facts_does_nothing(self):
        mock_add = MagicMock()
        with patch("therapy.database.add_profile_fact", mock_add):
            await save_extracted_facts("gma-1", "sess-1", [])
        mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_continues_after_individual_error(self):
        facts = [
            {"fact": "Eleanor grew roses", "confidence": 0.9},
            {"fact": "She loved baking", "confidence": 0.7},
        ]
        # First call raises, second succeeds
        mock_add = MagicMock(side_effect=[Exception("DB error"), None])
        with patch("therapy.database.add_profile_fact", mock_add):
            await save_extracted_facts("gma-1", "sess-1", facts)
        # Both were attempted
        assert mock_add.call_count == 2


# ---------------------------------------------------------------------------
# extract_and_save (orchestration)
# ---------------------------------------------------------------------------


class TestExtractAndSave:
    @pytest.mark.asyncio
    async def test_orchestrates_extract_then_save(self):
        facts = [{"fact": "Eleanor loved roses", "confidence": 0.9}]
        with (
            patch(
                "therapy.profile_extractor.extract_facts",
                new=AsyncMock(return_value=facts),
            ) as mock_extract,
            patch(
                "therapy.profile_extractor.save_extracted_facts",
                new=AsyncMock(),
            ) as mock_save,
        ):
            await extract_and_save(
                "gma-1", "sess-1", "I loved my roses.", make_memory(), make_history(), []
            )
        mock_extract.assert_awaited_once()
        mock_save.assert_awaited_once_with("gma-1", "sess-1", facts)

    @pytest.mark.asyncio
    async def test_swallows_errors_gracefully(self):
        with patch(
            "therapy.profile_extractor.extract_facts",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            # Should not raise
            await extract_and_save(
                "gma-1", "sess-1", "Hello.", make_memory(), [], []
            )

    @pytest.mark.asyncio
    async def test_passes_existing_facts_to_extract(self):
        existing = ["Eleanor grew roses in Pasadena"]
        with (
            patch(
                "therapy.profile_extractor.extract_facts",
                new=AsyncMock(return_value=[]),
            ) as mock_extract,
            patch("therapy.profile_extractor.save_extracted_facts", new=AsyncMock()),
        ):
            await extract_and_save(
                "gma-1", "sess-1", "msg", make_memory(), [], existing
            )
        _, kwargs = mock_extract.call_args
        # existing_facts passed as 4th positional arg
        assert mock_extract.await_args.args[3] == existing
