"""
HTTP fetching shared by the registry and page modules.

The registry's identity endpoint is mandatory: if it fails the caller has
no company to describe. Management and annual accounts are enrichments,
and Brreg occasionally resets a connection under load. A single blip
should cost a company its filed accounts, so those calls retry once
before degrading to an empty result that records why it is empty.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from src.request_tracker import request_tracker

DEFAULT_TIMEOUT = 20
DEFAULT_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 1.0


class NotFound(Exception):
    """The endpoint responded, but has no record for this key."""


def fetch_json(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    attempts: int = DEFAULT_ATTEMPTS,
    accept_missing: bool = False,
) -> Any:
    """GET ``url`` and decode the JSON body.

    Args:
        accept_missing: When ``True``, a 404 yields ``None`` instead of
            raising. Absence of a record is a legitimate answer for
            annual accounts and roles.

    Raises:
        NotFound: On 404 when ``accept_missing`` is ``False``.
        requests.RequestException: After all attempts fail.
    """
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            request_tracker.record()
            response = requests.get(
                url,
                timeout=timeout,
                headers={"Accept": "application/json"},
            )

            if response.status_code == 404:
                if accept_missing:
                    return None
                raise NotFound(f"404 Not Found: {url}")

            response.raise_for_status()

            # Brreg serves UTF-8 Norwegian text. Pinning the encoding stops
            # æ/ø/å being decoded as latin-1.
            response.encoding = "utf-8"
            return response.json()

        except NotFound:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(RETRY_BACKOFF_SECONDS)

    assert last_error is not None
    raise last_error


class ByteResponse:
    """A truncated, size-capped HTTP body."""

    def __init__(
        self,
        body: bytes,
        encoding: str,
        content_type: str,
        truncated: bool,
        status_code: int,
    ):
        self.body = body
        self.encoding = encoding
        self.content_type = content_type
        self.truncated = truncated
        self.status_code = status_code

    def text(self) -> str:
        return self.body.decode(self.encoding or "utf-8", errors="replace")


def fetch_bytes(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    attempts: int = DEFAULT_ATTEMPTS,
    max_bytes: int = 10_000_000,
    headers: dict[str, str] | None = None,
) -> ByteResponse:
    """GET ``url``, streaming and capping the body at ``max_bytes``.

    Streamed rather than buffered so an unexpectedly large or endless
    response cannot exhaust memory.
    """
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            request_tracker.record()
            with requests.get(
                url,
                timeout=timeout,
                headers=headers or {"Accept": "*/*"},
                stream=True,
            ) as response:
                response.raise_for_status()

                buffer = bytearray()
                truncated = False
                for chunk in response.iter_content(chunk_size=65536):
                    buffer.extend(chunk)
                    if len(buffer) >= max_bytes:
                        truncated = True
                        break

                return ByteResponse(
                    body=bytes(buffer[:max_bytes]),
                    encoding=response.encoding or "utf-8",
                    content_type=response.headers.get("Content-Type", ""),
                    truncated=truncated,
                    status_code=response.status_code,
                )

        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(RETRY_BACKOFF_SECONDS)

    assert last_error is not None
    raise last_error