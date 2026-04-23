"""Typed Pydantic models + thin Supabase query wrappers.

Field names mirror schema.sql exactly — this is the contract with Person B.
Import `sb` for the shared Supabase client; import the wrappers for common reads/writes.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------- models ----------
class Grandma(BaseModel):
    id: UUID
    name: str
    phone: str
    created_at: Optional[datetime] = None


class FamilyMember(BaseModel):
    id: UUID
    grandma_id: Optional[UUID] = None
    name: Optional[str] = None
    phone: str
    created_at: Optional[datetime] = None


class Memory(BaseModel):
    id: UUID
    grandma_id: Optional[UUID] = None
    submitted_by_family_id: Optional[UUID] = None
    image_url: Optional[str] = None
    original_caption: Optional[str] = None
    ai_summary: Optional[str] = None
    ai_tags: list[str] = Field(default_factory=list)
    people_mentioned: list[str] = Field(default_factory=list)
    emotion_hints: list[str] = Field(default_factory=list)
    era: Optional[str] = None
    used_in_sessions: list[Any] = Field(default_factory=list)
    created_at: Optional[datetime] = None


class MemoryInsert(BaseModel):
    grandma_id: UUID
    submitted_by_family_id: Optional[UUID] = None
    image_url: Optional[str] = None
    original_caption: Optional[str] = None
    ai_summary: Optional[str] = None
    ai_tags: list[str] = Field(default_factory=list)
    people_mentioned: list[str] = Field(default_factory=list)
    emotion_hints: list[str] = Field(default_factory=list)
    era: Optional[str] = None


class Session(BaseModel):
    id: UUID
    grandma_id: Optional[UUID] = None
    memory_id: Optional[UUID] = None
    status: str = "active"
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


class Turn(BaseModel):
    id: UUID
    session_id: Optional[UUID] = None
    role: Optional[str] = None  # "bot" | "grandma"
    content: Optional[str] = None
    image_url: Optional[str] = None
    created_at: Optional[datetime] = None


class TurnInsert(BaseModel):
    session_id: UUID
    role: str
    content: Optional[str] = None
    image_url: Optional[str] = None


class ProfileFact(BaseModel):
    id: UUID
    grandma_id: Optional[UUID] = None
    fact: str
    source_session_id: Optional[UUID] = None
    created_at: Optional[datetime] = None


class ProfileFactInsert(BaseModel):
    grandma_id: UUID
    fact: str
    source_session_id: Optional[UUID] = None


# ---------- wrappers ----------
def _first(rows: list[dict]) -> Optional[dict]:
    return rows[0] if rows else None


def get_grandma(grandma_id: str) -> Optional[Grandma]:
    r = sb.table("grandmas").select("*").eq("id", grandma_id).limit(1).execute()
    row = _first(r.data or [])
    return Grandma(**row) if row else None


def get_grandma_by_phone(phone: str) -> Optional[Grandma]:
    r = sb.table("grandmas").select("*").eq("phone", phone).limit(1).execute()
    row = _first(r.data or [])
    return Grandma(**row) if row else None


def get_family_by_phone(phone: str) -> Optional[FamilyMember]:
    r = sb.table("family_members").select("*").eq("phone", phone).limit(1).execute()
    row = _first(r.data or [])
    return FamilyMember(**row) if row else None


def list_memories(grandma_id: Optional[str] = None) -> list[Memory]:
    q = sb.table("memories").select("*").order("created_at", desc=True)
    if grandma_id:
        q = q.eq("grandma_id", grandma_id)
    r = q.execute()
    return [Memory(**row) for row in (r.data or [])]


def get_memory(memory_id: str) -> Optional[Memory]:
    r = sb.table("memories").select("*").eq("id", memory_id).limit(1).execute()
    row = _first(r.data or [])
    return Memory(**row) if row else None


def insert_memory(payload: MemoryInsert) -> Memory:
    r = sb.table("memories").insert(payload.model_dump(mode="json")).execute()
    return Memory(**r.data[0])


def get_active_session(grandma_id: str) -> Optional[Session]:
    r = (
        sb.table("sessions")
        .select("*")
        .eq("grandma_id", grandma_id)
        .eq("status", "active")
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    row = _first(r.data or [])
    return Session(**row) if row else None


def start_session(grandma_id: str, memory_id: str) -> Session:
    sb.table("sessions").update({"status": "ended"}).eq("grandma_id", grandma_id).eq(
        "status", "active"
    ).execute()
    r = (
        sb.table("sessions")
        .insert({"grandma_id": grandma_id, "memory_id": memory_id, "status": "active"})
        .execute()
    )
    return Session(**r.data[0])


def end_session(session_id: str) -> None:
    sb.table("sessions").update(
        {"status": "ended", "ended_at": datetime.utcnow().isoformat()}
    ).eq("id", session_id).execute()


def list_turns(session_id: str) -> list[Turn]:
    r = (
        sb.table("turns")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )
    return [Turn(**row) for row in (r.data or [])]


def insert_turn(payload: TurnInsert) -> Turn:
    r = sb.table("turns").insert(payload.model_dump(mode="json")).execute()
    return Turn(**r.data[0])


def list_profile_facts(grandma_id: str) -> list[ProfileFact]:
    r = (
        sb.table("grandma_profile_facts")
        .select("*")
        .eq("grandma_id", grandma_id)
        .order("created_at", desc=True)
        .execute()
    )
    return [ProfileFact(**row) for row in (r.data or [])]


def insert_profile_fact(payload: ProfileFactInsert) -> ProfileFact:
    r = sb.table("grandma_profile_facts").insert(payload.model_dump(mode="json")).execute()
    return ProfileFact(**r.data[0])
