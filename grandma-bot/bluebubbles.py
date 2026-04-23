"""BlueBubbles client + inbound payload parser."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel

BLUEBUBBLES_URL = os.getenv("BLUEBUBBLES_URL", "http://localhost:1234").rstrip("/")
BLUEBUBBLES_PASSWORD = os.getenv("BLUEBUBBLES_PASSWORD", "")

# NOTE: shared sync client — fine for hackathon traffic volumes.
_client = httpx.Client(timeout=30.0)


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
        json={"chatGuid": chat_guid, "message": message, "method": method},
    )


def send_text(phone: str, text: str) -> None:
    """Send an iMessage text. Falls back from private-api to apple-script."""
    chat_guid = chat_guid_for(phone)
    try:
        r = _send_text_with_method(chat_guid, text, "private-api")
        if r.status_code >= 400:
            r = _send_text_with_method(chat_guid, text, "apple-script")
        r.raise_for_status()
    except Exception as e:
        print(f"[bluebubbles] send_text failed to {phone}: {e}")


def send_image(phone: str, image_url_or_path: str, caption: Optional[str] = None) -> None:
    """Send an iMessage attachment. Accepts a local filesystem path or an http(s) URL."""
    chat_guid = chat_guid_for(phone)
    url = f"{BLUEBUBBLES_URL}/api/v1/message/attachment"

    try:
        if image_url_or_path.startswith("http://") or image_url_or_path.startswith("https://"):
            img_bytes = _client.get(image_url_or_path).content
            name = image_url_or_path.rsplit("/", 1)[-1].split("?")[0] or "memory.jpg"
        else:
            p = Path(image_url_or_path)
            img_bytes = p.read_bytes()
            name = p.name

        files = {"attachment": (name, img_bytes, "image/jpeg")}
        data = {"chatGuid": chat_guid, "name": name, "method": "private-api"}
        if caption:
            data["message"] = caption

        r = _client.post(url, params={"password": BLUEBUBBLES_PASSWORD}, data=data, files=files)
        if r.status_code >= 400:
            data["method"] = "apple-script"
            r = _client.post(url, params={"password": BLUEBUBBLES_PASSWORD}, data=data, files=files)
        r.raise_for_status()

        if caption:
            # NOTE: some BlueBubbles builds ignore the multipart `message` field — send as a follow-up.
            send_text(phone, caption)
    except Exception as e:
        print(f"[bluebubbles] send_image failed to {phone}: {e}")


def download_attachment(guid: str) -> bytes:
    """Fetch raw bytes for an inbound attachment."""
    url = f"{BLUEBUBBLES_URL}/api/v1/attachment/{guid}/download"
    r = _client.get(url, params={"password": BLUEBUBBLES_PASSWORD})
    r.raise_for_status()
    return r.content
