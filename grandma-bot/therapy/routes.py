"""FastAPI router for therapy session management APIs.

Mount in main.py:
    from therapy.routes import router as therapy_router
    app.include_router(therapy_router)

Endpoints
---------
POST /api/therapy/start-session        — start a session (auto-selects memory if omitted)
GET  /api/therapy/session/{grandma_id} — active session with turns
GET  /api/therapy/profile/{grandma_id} — all profile facts for a grandma
POST /api/therapy/end-session/{session_id} — manually end a session
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import database as db
from .handler import select_next_memory, start_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/therapy", tags=["therapy"])


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class StartSessionRequest(BaseModel):
    grandma_id: str
    memory_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/start-session")
async def api_start_session(body: StartSessionRequest):
    """Start a therapy session for a grandma.

    If *memory_id* is omitted, the best unused memory is auto-selected.
    Returns 409 if an active session already exists, 404 if grandma/memory
    is not found, 200 with the new session row on success.
    """
    grandma_id = body.grandma_id

    # Validate grandma exists
    try:
        grandma = await asyncio.to_thread(db.get_grandma_by_id, grandma_id)
    except Exception as exc:
        logger.error("[routes] start-session: DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    if grandma is None:
        raise HTTPException(status_code=404, detail="Grandma not found")

    # Guard: active session already running
    try:
        existing = await asyncio.to_thread(db.get_active_session, grandma_id)
    except Exception as exc:
        logger.error("[routes] start-session: active session check failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    if existing is not None:
        return JSONResponse(
            status_code=409,
            content={"error": "Active session already exists", "session_id": existing["id"]},
        )

    # Resolve memory_id
    memory_id = body.memory_id
    if not memory_id:
        try:
            memory_id = await select_next_memory(grandma_id)
        except Exception as exc:
            logger.error("[routes] start-session: select_next_memory failed: %s", exc)
            raise HTTPException(status_code=500, detail="Could not select memory")

        if not memory_id:
            raise HTTPException(
                status_code=404,
                detail="No unused memories available for this grandma",
            )

    # Validate memory exists
    try:
        memory = await asyncio.to_thread(db.get_memory, memory_id)
    except Exception as exc:
        logger.error("[routes] start-session: get_memory failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    # Kick off the session (non-blocking background task so HTTP returns fast)
    asyncio.create_task(start_session(grandma_id, memory_id))

    return {
        "status": "starting",
        "grandma_id": grandma_id,
        "memory_id": memory_id,
        "message": "Session is being started — opener will be sent to grandma shortly.",
    }


@router.get("/session/{grandma_id}")
async def api_get_session(grandma_id: str):
    """Return the active session for *grandma_id*, including all turns.

    Returns 404 if no active session exists.
    """
    try:
        session = await asyncio.to_thread(db.get_active_session, grandma_id)
    except Exception as exc:
        logger.error("[routes] get-session: DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    if session is None:
        raise HTTPException(status_code=404, detail="No active session")

    try:
        turns = await asyncio.to_thread(db.get_session_turns, session["id"])
    except Exception as exc:
        logger.warning("[routes] get-session: turns fetch failed: %s", exc)
        turns = []

    session["turns"] = turns
    return session


@router.get("/profile/{grandma_id}")
async def api_get_profile(grandma_id: str):
    """Return all profile facts learned about *grandma_id* across sessions."""
    try:
        facts = await asyncio.to_thread(db.get_profile_facts, grandma_id)
    except Exception as exc:
        logger.error("[routes] get-profile: DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    return facts


@router.post("/end-session/{session_id}")
async def api_end_session(session_id: str):
    """Manually end a session (e.g. from the dashboard).

    Idempotent — ending an already-ended session is a no-op.
    """
    try:
        await asyncio.to_thread(db.end_session, session_id)
    except Exception as exc:
        logger.error("[routes] end-session: DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    return {"status": "ended", "session_id": session_id}
