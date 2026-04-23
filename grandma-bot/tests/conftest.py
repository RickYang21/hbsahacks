"""Shared pytest configuration.

Injects stub environment variables so pydantic-settings Settings() can be
instantiated without a real .env file.  Tests that need real credentials
should be marked @pytest.mark.integration and skipped in CI.
"""
import os
import sys
import types
from unittest.mock import MagicMock

# ── 1. Env vars ──────────────────────────────────────────────────────────────
# Set before any therapy.* imports so pydantic-settings validation passes.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

# ── 2. Stub out heavy optional deps not installed in the test venv ────────────
# supabase-py — stub the package so therapy.database can be imported cleanly.
_supabase_stub = types.ModuleType("supabase")
_supabase_stub.Client = MagicMock()
_supabase_stub.create_client = MagicMock()
sys.modules.setdefault("supabase", _supabase_stub)

# therapy.database — expose stub functions so tests can patch them without
# needing a real Supabase connection.
_db_stub = types.ModuleType("therapy.database")
for _fn in (
    "get_grandma_by_phone", "get_active_session", "create_session",
    "end_session", "add_turn", "get_session_turns", "get_memory",
    "get_unused_memories", "mark_memory_used", "add_profile_fact",
    "get_profile_facts",
):
    setattr(_db_stub, _fn, MagicMock())
sys.modules.setdefault("therapy.database", _db_stub)
