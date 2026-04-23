"""Grandma Memory Bot — intake pipeline, shared DB, dashboard API."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from supabase import create_client, Client

import bluebubbles
from bluebubbles import Inbound, parse_inbound

# ---------- config ----------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

STORAGE_BUCKET = "memory-photos"
CLAUDE_MODEL = "claude-sonnet-4-5"
MERGE_WINDOW_SECONDS = 60

# NOTE: hardcoded single-grandma — matches seed in schema.sql.
ELEANOR_ID = "11111111-1111-1111-1111-111111111111"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ---------- app ----------
app = FastAPI(title="Grandma Memory Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 60s merge buffer (in-memory) ----------
# NOTE: in-memory dict keyed by sender phone — fine for hackathon, lost on restart.
_pending_text: dict[str, tuple[str, float]] = {}


def _stash_text(phone: str, text: str) -> None:
    _pending_text[phone] = (text, time.time())


def _pop_recent_text(phone: str) -> Optional[str]:
    entry = _pending_text.get(phone)
    if not entry:
        return None
    text, ts = entry
    if time.time() - ts > MERGE_WINDOW_SECONDS:
        _pending_text.pop(phone, None)
        return None
    _pending_text.pop(phone, None)
    return text


# ---------- helpers ----------
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = _JSON_FENCE_RE.sub("", s)
    s = s.strip()
    # Also handle ```json ... ``` where the fences aren't at absolute edges.
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    return s.strip()


def _lookup_family(phone: str) -> Optional[dict]:
    r = (
        supabase.table("family_members")
        .select("*, grandmas(*)")
        .eq("phone", phone)
        .limit(1)
        .execute()
    )
    return r.data[0] if r.data else None


def _lookup_grandma(phone: str) -> Optional[dict]:
    r = supabase.table("grandmas").select("*").eq("phone", phone).limit(1).execute()
    return r.data[0] if r.data else None


def _upload_photo(image_bytes: bytes) -> str:
    key = f"{uuid.uuid4()}.jpg"
    supabase.storage.from_(STORAGE_BUCKET).upload(
        path=key,
        file=image_bytes,
        file_options={"content-type": "image/jpeg", "upsert": "true"},
    )
    return supabase.storage.from_(STORAGE_BUCKET).get_public_url(key)


def _analyze_with_claude(image_url: str, caption: Optional[str]) -> dict:
    prompt = (
        "You are tagging a family photo a loved one is sharing with an elderly relative who has memory loss. "
        f'Caption from the sender: "{caption or ""}".\n'
        "Return ONLY a JSON object — no prose, no code fences — with exactly these keys:\n"
        '  "summary": one warm sentence describing the photo.\n'
        '  "tags": array of 5-10 short lowercase strings.\n'
        '  "people_mentioned": array of names or relationships referenced.\n'
        '  "emotion_hints": array like ["warmth","pride","nostalgia"].\n'
        '  "era": string like "1980s" or "unknown".'
    )
    resp = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except Exception:
        print(f"[claude] failed to parse JSON. raw output:\n{raw}")
        raise
    return {
        "summary": data.get("summary") or "",
        "tags": data.get("tags") or [],
        "people_mentioned": data.get("people_mentioned") or [],
        "emotion_hints": data.get("emotion_hints") or [],
        "era": data.get("era") or "unknown",
    }


# ---------- handlers ----------
def handle_family(family: dict, msg: Inbound) -> dict:
    grandma = family.get("grandmas") or {}
    grandma_name = grandma.get("name") or "her"
    family_phone = msg.from_phone

    # Text only — stash for merge.
    if not msg.attachment_guids:
        if msg.text:
            _stash_text(family_phone, msg.text)
        return {"status": "buffered_text"}

    caption = msg.text or _pop_recent_text(family_phone)
    saved = []

    for guid in msg.attachment_guids:
        try:
            img_bytes = bluebubbles.download_attachment(guid)
            public_url = _upload_photo(img_bytes)
            extracted = _analyze_with_claude(public_url, caption)

            row = {
                "grandma_id": grandma["id"],
                "submitted_by_family_id": family["id"],
                "image_url": public_url,
                "original_caption": caption,
                "ai_summary": extracted["summary"],
                "ai_tags": extracted["tags"],
                "people_mentioned": extracted["people_mentioned"],
                "emotion_hints": extracted["emotion_hints"],
                "era": extracted["era"],
            }
            ins = supabase.table("memories").insert(row).execute()
            saved.append(ins.data[0] if ins.data else row)

            confirmation = (
                f'Got it — saved "{extracted["summary"]}". '
                f"I'll share it with {grandma_name} this week ❤️"
            )
            bluebubbles.send_text(family_phone, confirmation)
        except Exception as e:
            print(f"[family_handler] failure on guid={guid}: {e}")
            bluebubbles.send_text(family_phone, "Something broke on my end — try again?")

    return {"status": "ok", "saved_count": len(saved)}


def handle_grandma(grandma: dict, msg: Inbound) -> dict:
    # TODO: Person B implements here — reminiscence therapy conversation engine.
    print(f"[grandma_handler stub] grandma_id={grandma['id']} text={msg.text!r}")
    return {"status": "routed_to_grandma_handler", "grandma_id": grandma["id"]}


# ---------- routes ----------
@app.post("/webhook/bluebubbles")
async def bluebubbles_webhook(request: Request):
    payload = await request.json()
    msg = parse_inbound(payload)
    if msg is None:
        return {"status": "ignored"}

    family = _lookup_family(msg.from_phone)
    if family:
        return handle_family(family, msg)

    grandma = _lookup_grandma(msg.from_phone)
    if grandma:
        return handle_grandma(grandma, msg)

    print(f"[webhook] unknown sender {msg.from_phone} — ignoring")
    return {"status": "unknown_sender"}


@app.get("/api/memories")
def api_memories():
    r = (
        supabase.table("memories")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return r.data or []


@app.get("/api/profile/{grandma_id}")
def api_profile(grandma_id: str):
    r = (
        supabase.table("grandma_profile_facts")
        .select("*")
        .eq("grandma_id", grandma_id)
        .order("created_at", desc=True)
        .execute()
    )
    return r.data or []


@app.get("/api/sessions/active")
def api_active_session():
    r = (
        supabase.table("sessions")
        .select("*, memories(*)")
        .eq("status", "active")
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if not r.data:
        return JSONResponse(content=None)
    session = r.data[0]
    turns = (
        supabase.table("turns")
        .select("*")
        .eq("session_id", session["id"])
        .order("created_at", desc=True)
        .execute()
    )
    session["turns"] = list(reversed(turns.data or []))
    return session


@app.post("/api/session/start/{memory_id}")
def api_session_start(memory_id: str):
    mem_r = supabase.table("memories").select("*").eq("id", memory_id).limit(1).execute()
    if not mem_r.data:
        return JSONResponse(status_code=404, content={"error": "memory not found"})
    memory = mem_r.data[0]

    g_r = supabase.table("grandmas").select("*").eq("id", memory["grandma_id"]).limit(1).execute()
    grandma = g_r.data[0] if g_r.data else None
    if not grandma:
        return JSONResponse(status_code=404, content={"error": "grandma not found"})

    # Mark any previously-active sessions for this grandma as ended (demo hygiene).
    supabase.table("sessions").update({"status": "ended"}).eq(
        "grandma_id", grandma["id"]
    ).eq("status", "active").execute()

    submitter_name = "the family"
    if memory.get("submitted_by_family_id"):
        fam = (
            supabase.table("family_members")
            .select("name")
            .eq("id", memory["submitted_by_family_id"])
            .limit(1)
            .execute()
        )
        if fam.data and fam.data[0].get("name"):
            submitter_name = fam.data[0]["name"]

    opener = f"Hi Mom 💐 Look what {submitter_name} sent me today… does this look familiar?"

    sess_r = (
        supabase.table("sessions")
        .insert(
            {
                "grandma_id": grandma["id"],
                "memory_id": memory["id"],
                "status": "active",
            }
        )
        .execute()
    )
    session = sess_r.data[0]

    # Send photo + opener to grandma via iMessage.
    if memory.get("image_url"):
        bluebubbles.send_image(grandma["phone"], memory["image_url"], caption=opener)
    else:
        bluebubbles.send_text(grandma["phone"], opener)

    supabase.table("turns").insert(
        {
            "session_id": session["id"],
            "role": "bot",
            "content": opener,
            "image_url": memory.get("image_url"),
        }
    ).execute()

    return session


@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "dashboard.html"))


@app.get("/healthz")
def healthz():
    return {"ok": True, "grandma_id": ELEANOR_ID}
