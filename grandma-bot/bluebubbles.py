"""BlueBubbles client + inbound payload parser."""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel

BLUEBUBBLES_URL = os.getenv("BLUEBUBBLES_URL", "http://localhost:1234").rstrip("/")
BLUEBUBBLES_PASSWORD = os.getenv("BLUEBUBBLES_PASSWORD", "")

# NOTE: shared sync client — fine for hackathon traffic volumes.
# http2=False: BlueBubbles doesn't support HTTP/2; h2 is installed via supabase deps.
_client = httpx.Client(timeout=30.0, http2=False)
_image_client = httpx.Client(timeout=120.0, http2=False)


class Inbound(BaseModel):
    from_phone: str
    text: Optional[str] = None
    attachment_guids: list[str] = []


def _normalize_phone(raw: str) -> str:
    """Strip spaces/dashes, keep leading + if present. Email addresses pass through."""
    if not raw:
        return raw
    if "@" in raw:
        return raw.strip().lower()
    cleaned = raw.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if not cleaned.startswith("+") and cleaned.isdigit():
        # NOTE: assume US if no country code — fine for hackathon demo.
        if len(cleaned) == 10:
            cleaned = "+1" + cleaned
        elif len(cleaned) == 11 and cleaned.startswith("1"):
            cleaned = "+" + cleaned
        else:
            cleaned = "+" + cleaned
    return cleaned


def chat_guid_for(phone: str) -> str:
    return f"iMessage;-;{phone}"


def parse_inbound(payload: dict) -> Optional[Inbound]:
    """Normalize a BlueBubbles webhook event into Inbound, or None to ignore."""
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "new-message":
        return None
    data = payload.get("data") or {}
    if data.get("isFromMe"):
        return None

    handle = data.get("handle") or {}
    from_phone = handle.get("address") or ""
    if not from_phone:
        return None
    from_phone = _normalize_phone(from_phone)

    text = data.get("text") or None
    if text is not None:
        text = text.strip() or None

    attachments = data.get("attachments") or []
    attachment_guids: list[str] = []
    for a in attachments:
        guid = a.get("guid")
        mime = (a.get("mimeType") or "").lower()
        # Only forward image attachments; ignore stickers / other types.
        if guid and mime.startswith("image/"):
            attachment_guids.append(guid)

    if text is None and not attachment_guids:
        return None

    return Inbound(from_phone=from_phone, text=text, attachment_guids=attachment_guids)


def _send_text_with_method(chat_guid: str, message: str, method: str) -> httpx.Response:
    url = f"{BLUEBUBBLES_URL}/api/v1/message/text"
    return _client.post(
        url,
        params={"password": BLUEBUBBLES_PASSWORD},
        json={"chatGuid": chat_guid, "message": message, "method": method, "tempGuid": f"temp-{uuid.uuid4()}"},
 )


def send_text(phone: str, text: str) -> None:
    """Send an iMessage text. Uses apple-script (private-api not required)."""
    chat_guid = chat_guid_for(phone)
    try:
        r = _send_text_with_method(chat_guid, text, "apple-script")
        if r.status_code >= 400:
            print(f"[bluebubbles] send_text apple-script {r.status_code}: {r.text}")
            r = _send_text_with_method(chat_guid, text, "private-api")
        if r.status_code >= 400:
            print(f"[bluebubbles] send_text private-api {r.status_code}: {r.text}")
        r.raise_for_status()
    except Exception as e:
        print(f"[bluebubbles] send_text failed to {phone}: {e}")


def send_image(phone: str, image_url_or_path: str, caption: Optional[str] = None) -> None:
    """Send an iMessage attachment. Accepts a local filesystem path or an http(s) URL."""
    chat_guid = chat_guid_for(phone)
    url = f"{BLUEBUBBLES_URL}/api/v1/message/attachment"

    try:
        if image_url_or_path.startswith("http://") or image_url_or_path.startswith("https://"):
            print(f"[bluebubbles] downloading image from {image_url_or_path[:80]}")
            img_bytes = _image_client.get(image_url_or_path).content
            print(f"[bluebubbles] downloaded {len(img_bytes)} bytes")
            name = image_url_or_path.rsplit("/", 1)[-1].split("?")[0] or "memory.jpg"
        else:
            p = Path(image_url_or_path)
            img_bytes = p.read_bytes()
            name = p.name

        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".heic")):
            name = name + ".jpg"

        files = {"attachment": (name, img_bytes, "image/jpeg")}
        data = {
            "chatGuid": chat_guid,
            "name": name,
            "method": "apple-script",
            "tempGuid": f"temp-{uuid.uuid4()}",
        }
        if caption:
            data["message"] = caption

        print(f"[bluebubbles] uploading attachment to BlueBubbles ({len(img_bytes)} bytes)")
        r = _image_client.post(url, params={"password": BLUEBUBBLES_PASSWORD}, data=data, files=files)
        print(f"[bluebubbles] attachment response {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            data["method"] = "private-api"
            data["tempGuid"] = f"temp-{uuid.uuid4()}"
            r = _image_client.post(url, params={"password": BLUEBUBBLES_PASSWORD}, data=data, files=files)
            print(f"[bluebubbles] attachment fallback response {r.status_code}: {r.text[:200]}")
        r.raise_for_status()

        if caption:
            send_text(phone, caption)
    except Exception as e:
        print(f"[bluebubbles] send_image failed to {phone}: {e}")


def download_attachment(guid: str) -> bytes:
    """Fetch raw bytes for an inbound attachment."""
    url = f"{BLUEBUBBLES_URL}/api/v1/attachment/{guid}/download"
    r = _client.get(url, params={"password": BLUEBUBBLES_PASSWORD})
    r.raise_for_status()
    return r.content
