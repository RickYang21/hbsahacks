"""Async BlueBubbles REST client for the therapy conversation engine."""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import bluebubbles as _root_bb  # root client — used to share the echo-dedup registry

from .config import settings

_TIMEOUT = httpx.Timeout(30.0)
_RETRY_ERRORS = (httpx.TransportError, httpx.TimeoutException)

# Send-level dedup: refuse to send the same text to the same phone twice within
# this window. Catches echo loops, concurrent handler races, fallback-method
# double-sends, and retry-after-success scenarios.
_SEND_DEDUP_TTL = 120.0
_recent_sends: dict[str, tuple[str, float]] = {}


def _normalize(text: str) -> str:
    """Normalize text for dedup: collapse whitespace, strip punctuation padding."""
    return " ".join(text.split()).strip()


def _is_duplicate_send(phone: str, text: str) -> bool:
    """Return True if *text* was already sent to *phone* recently."""
    now = time.time()
    key = _normalize(text)
    expired = [p for p, (_, t) in list(_recent_sends.items()) if now - t > _SEND_DEDUP_TTL]
    for p in expired:
        _recent_sends.pop(p, None)
    recent = _recent_sends.get(phone)
    if recent and recent[0] == key and now - recent[1] < _SEND_DEDUP_TTL:
        return True
    _recent_sends[phone] = (key, now)
    return False


def _chat_guid(phone: str) -> str:
    return f"iMessage;-;{phone}"


def _params() -> dict:
    return {"password": settings.bluebubbles_password}


def _base() -> str:
    return settings.bluebubbles_url.rstrip("/")


class BlueBubblesClient:
    """Async wrapper around the BlueBubbles REST API."""

    def __init__(self, base_url: str | None = None, password: str | None = None) -> None:
        self._base = (base_url or settings.bluebubbles_url).rstrip("/")
        self._password = password if password is not None else settings.bluebubbles_password
        # http2=False: BlueBubbles doesn't support HTTP/2; h2 installed via supabase deps.
        self._client = httpx.AsyncClient(timeout=_TIMEOUT, http2=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _params(self) -> dict:
        return {"password": self._password}

    def _chat_guid(self, phone: str) -> str:
        return f"iMessage;-;{phone}"

    @retry(
        retry=retry_if_exception_type(_RETRY_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _post_text(self, chat_guid: str, message: str, method: str) -> httpx.Response:
        return await self._client.post(
            f"{self._base}/api/v1/message/text",
            params=self._params(),
            json={"chatGuid": chat_guid, "message": message, "method": method, "tempGuid": f"temp-{uuid.uuid4()}"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_text(self, phone: str, message: str) -> dict:
        """Send a text iMessage. Uses apple-script (private-api not required)."""
        if _is_duplicate_send(phone, message):
            print(f"[therapy.bb] DEDUP: dropping duplicate send to {phone}: {message[:60]!r}")
            return {"status": "dedup_skipped"}
        _root_bb._sent_texts[_root_bb._normalize(message)] = time.time()  # register before sending so echo-dedup catches it
        chat_guid = self._chat_guid(phone)
        try:
            r = await self._post_text(chat_guid, message, "apple-script")
            if r.status_code >= 400:
                r = await self._post_text(chat_guid, message, "private-api")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            print(f"[therapy.bb] send_text to {phone} failed: {exc}")
            return {"error": str(exc)}

    async def send_image(
        self, phone: str, image_url: str, caption: Optional[str] = None
    ) -> dict:
        """Download *image_url* then send as an iMessage attachment.

        If *caption* is provided it is sent as a separate follow-up text
        (more reliable than the multipart `message` field).
        """
        chat_guid = self._chat_guid(phone)
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as dl:
                img_resp = await dl.get(image_url)
                img_resp.raise_for_status()
                img_bytes = img_resp.content

            name = image_url.rsplit("/", 1)[-1].split("?")[0] or "memory.jpg"
            files = {"attachment": (name, img_bytes, "image/jpeg")}
            data = {"chatGuid": chat_guid, "name": name, "method": "private-api"}

            @retry(
                retry=retry_if_exception_type(_RETRY_ERRORS),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=8),
                reraise=True,
            )
            async def _upload() -> httpx.Response:
                return await self._client.post(
                    f"{self._base}/api/v1/message/attachment",
                    params=self._params(),
                    data=data,
                    files=files,
                )

            r = await _upload()
            if r.status_code >= 400:
                data["method"] = "apple-script"
                r = await _upload()
            r.raise_for_status()
            result = r.json()
        except Exception as exc:
            print(f"[therapy.bb] send_image to {phone} failed: {exc}")
            result = {"error": str(exc)}

        if caption:
            await asyncio.sleep(0.5)  # small gap so image arrives before caption
            await self.send_text(phone, caption)

        return result

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "BlueBubblesClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


# Module-level default instance — reused across requests (keeps connection pool alive).
client = BlueBubblesClient()
