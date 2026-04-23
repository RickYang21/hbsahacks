"""Supabase helpers scoped to the therapy conversation engine.

All functions are synchronous (supabase-py uses sync I/O under the hood).
Call them from async FastAPI routes via asyncio.to_thread() if needed.
"""
from __future__ import annotations

from typing import Optional

from supabase import Client, create_client

from .config import settings

_supabase: Optional[Client] = None


def _db() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(settings.supabase_url, settings.effective_supabase_key)
    return _supabase


# ---------------------------------------------------------------------------
# Grandma & session helpers
# ---------------------------------------------------------------------------


def get_grandma_by_phone(phone: str) -> dict | None:
    r = _db().table("grandmas").select("*").eq("phone", phone).limit(1).execute()
    return r.data[0] if r.data else None


def get_grandma_by_id(grandma_id: str) -> dict | None:
    r = _db().table("grandmas").select("*").eq("id", grandma_id).limit(1).execute()
    return r.data[0] if r.data else None


def get_active_session(grandma_id: str) -> dict | None:
    r = (
        _db()
        .table("sessions")
        .select("*, memories(*)")
        .eq("grandma_id", grandma_id)
        .eq("status", "active")
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    return r.data[0] if r.data else None


def create_session(grandma_id: str, memory_id: str) -> dict:
    r = (
        _db()
        .table("sessions")
        .insert({"grandma_id": grandma_id, "memory_id": memory_id, "status": "active"})
        .execute()
    )
    return r.data[0]


def get_last_ended_session(grandma_id: str) -> dict | None:
    """Return the most recently ended session for *grandma_id*, with memory joined."""
    r = (
        _db()
        .table("sessions")
        .select("*, memories(*)")
        .eq("grandma_id", grandma_id)
        .eq("status", "ended")
        .order("ended_at", desc=True)
        .limit(1)
        .execute()
    )
    return r.data[0] if r.data else None


def end_session(session_id: str) -> None:
    from datetime import datetime, timezone

    _db().table("sessions").update(
        {"status": "ended", "ended_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", session_id).execute()


def flag_session(session_id: str) -> None:
    """Mark session as flagged (distress/confusion detected) without ending it."""
    from datetime import datetime, timezone

    _db().table("sessions").update(
        {"status": "flagged", "ended_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", session_id).execute()


def add_session_alert(
    session_id: str,
    alert_type: str,
    grandma_message: str,
) -> dict:
    """Insert a row into session_alerts for dashboard visibility."""
    r = (
        _db()
        .table("session_alerts")
        .insert({
            "session_id": session_id,
            "alert_type": alert_type,
            "grandma_message": grandma_message,
        })
        .execute()
    )
    return r.data[0]


def get_family_members(grandma_id: str) -> list[dict]:
    """Return all family members registered for *grandma_id*."""
    r = (
        _db()
        .table("family_members")
        .select("*")
        .eq("grandma_id", grandma_id)
        .execute()
    )
    return r.data or []


def get_past_grandma_turns(grandma_id: str, limit: int = 60) -> list[str]:
    """Return grandma's message texts from all previously ended sessions.

    Used for repeated-story detection — does NOT include the current session.
    """
    sessions_r = (
        _db()
        .table("sessions")
        .select("id")
        .eq("grandma_id", grandma_id)
        .eq("status", "ended")
        .execute()
    )
    session_ids = [s["id"] for s in (sessions_r.data or [])]
    if not session_ids:
        return []

    r = (
        _db()
        .table("turns")
        .select("content")
        .in_("session_id", session_ids)
        .eq("role", "grandma")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return [t["content"] for t in (r.data or []) if t.get("content")]


# ---------------------------------------------------------------------------
# Turn helpers
# ---------------------------------------------------------------------------


def add_turn(
    session_id: str,
    role: str,
    content: str,
    image_url: str | None = None,
) -> dict:
    row: dict = {"session_id": session_id, "role": role, "content": content}
    if image_url:
        row["image_url"] = image_url
    r = _db().table("turns").insert(row).execute()
    return r.data[0]


def get_session_turns(session_id: str) -> list[dict]:
    r = (
        _db()
        .table("turns")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return r.data or []


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def get_memory(memory_id: str) -> dict | None:
    r = _db().table("memories").select("*").eq("id", memory_id).limit(1).execute()
    return r.data[0] if r.data else None


def get_unused_memories(grandma_id: str) -> list[dict]:
    """Return memories for *grandma_id* whose used_in_sessions array is empty."""
    r = (
        _db()
        .table("memories")
        .select("*")
        .eq("grandma_id", grandma_id)
        .order("created_at", desc=True)
        .execute()
    )
    all_memories = r.data or []
    # Filter client-side: unused = empty json array or null.
    return [m for m in all_memories if not (m.get("used_in_sessions") or [])]


def mark_memory_used(memory_id: str, session_id: str) -> None:
    """Append *session_id* to the memory's used_in_sessions json array."""
    mem = get_memory(memory_id)
    if mem is None:
        return
    current: list = mem.get("used_in_sessions") or []
    if session_id not in current:
        current.append(session_id)
    _db().table("memories").update({"used_in_sessions": current}).eq("id", memory_id).execute()


# ---------------------------------------------------------------------------
# Profile fact helpers
# ---------------------------------------------------------------------------


def add_profile_fact(
    grandma_id: str,
    fact: str,
    session_id: str,
    confidence: float = 1.0,  # schema has no confidence column; accepted for API compat
) -> dict:
    """Persist a new fact learned about grandma during a session.

    Note: the current schema does not have a *confidence* column, so that
    argument is accepted but not stored.
    """
    r = (
        _db()
        .table("grandma_profile_facts")
        .insert({"grandma_id": grandma_id, "fact": fact, "source_session_id": session_id})
        .execute()
    )
    return r.data[0]


def get_profile_facts(grandma_id: str) -> list[dict]:
    r = (
        _db()
        .table("grandma_profile_facts")
        .select("*")
        .eq("grandma_id", grandma_id)
        .order("created_at", desc=True)
        .execute()
    )
    return r.data or []
