"""
HTTP fetching shared by the registry and page modules.

The registry's identity endpoint is mandatory: if it fails the caller has
no company to describe. Management and annual accounts are enrichments,
and Brreg occasionally resets a connection under load. A single blip
should cost a company its filed accounts, so those calls retry once
before degrading to an empty result that records why it is empty.
"""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import urljoin

import requests

from src.hostreputation import reputation
from src.netguard import (
    DEFAULT_MAX_REDIRECTS,
    TooManyRedirects,
    UnsafeUrl,
    redirect_target,
    validate_url,
)
from src.request_tracker import request_tracker

DEFAULT_TIMEOUT = int(os.getenv("AGENT_HTTP_TIMEOUT", "20"))
DEFAULT_ATTEMPTS = int(os.getenv("AGENT_HTTP_ATTEMPTS", "2"))
RETRY_BACKOFF_SECONDS = 1.0


class NotFound(Exception):
    """The endpoint responded, but has no record for this key."""


class HostRefused(Exception):
    """This host has refused us enough times to stop asking."""


def get_following_redirects(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    headers: dict[str, str] | None = None,
    stream: bool = False,
    validate: bool = True,
    track_reputation: bool | None = None,
):
    """GET ``url``, validating every redirect hop.

    Redirects are followed by hand rather than by ``requests`` because
    the library does not re-check them. A URL that passed the private
    address guard can redirect to one that did not, so a discovered
    ``https://example.com`` becomes ``http://169.254.169.254/`` and
    instance metadata is read. Validating only the URL the caller
    supplied guards the request that was never sent.

    Args:
        validate: When ``False`` the address guard is skipped. Only the
            registry sets this, for a fixed set of Brreg endpoints that
            cannot be pointed anywhere by a search result.
        track_reputation: Defaults to following ``validate``. Reputation
            tracks hosts we have no prior relationship with, so a fixed
            Brreg endpoint is never worth a strike -- it is either
            always available or genuinely broken, and neither is
            something a strike count improves.

    Raises:
        UnsafeUrl: Any hop resolves to a private address or is malformed.
        TooManyRedirects: The chain exceeded ``max_redirects``.
        HostRefused: The host has refused automated access often enough
            that asking again is known to be wasted.
    """
    if track_reputation is None:
        track_reputation = validate

    current = validate_url(url) if validate else url
    remaining = max_redirects

    while True:
        if track_reputation and reputation.should_skip(current):
            raise HostRefused(
                f"{reputation.host_of(current)} refuses automated access; "
                "not requesting again in this run"
            )

        response = requests.get(
            current,
            timeout=timeout,
            headers=headers,
            stream=stream,
            # Handled here, so the library must not also follow them.
            allow_redirects=False,
        )

        if track_reputation:
            # Recorded before the redirect branch, so a refusal on any
            # hop is remembered against the host that served it.
            reputation.record(current, response.status_code)

        if response.status_code in (301, 302, 303, 307, 308):
            location = redirect_target(response, max_redirects=remaining)
            if location is None:
                return response

            remaining -= 1
            next_url = urljoin(current, location)

            # Every hop is re-validated, which is the entire reason this
            # function exists.
            if validate:
                next_url = validate_url(next_url)

            response.close()
            current = next_url
            continue

        return response


def fetch_json(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    attempts: int = DEFAULT_ATTEMPTS,
    accept_missing: bool = False,
    headers: dict[str, str] | None = None,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    validate: bool = True,
) -> Any:
    """GET ``url`` and decode the JSON body.

    Args:
        accept_missing: When ``True``, a 404 yields ``None`` instead of
            raising. Absence of a record is a legitimate answer for
            annual accounts and roles.
        headers: Extra headers merged over the default ``Accept``. The
            search API authenticates this way; credentials are never
            placed in the URL, because URLs get written to the cache.
        validate: Whether to apply the private-address guard to every
            hop. Off only for fixed registry endpoints.

    Raises:
        NotFound: On 404 when ``accept_missing`` is ``False``.
        requests.RequestException: After all attempts fail.
        UnsafeUrl: A hop points somewhere internal.
    """
    last_error: Exception | None = None

    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    for attempt in range(attempts):
        try:
            request_tracker.record()

            response = get_following_redirects(
                url,
                timeout=timeout,
                max_redirects=max_redirects,
                headers=request_headers,
                validate=validate,
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

        except (NotFound, UnsafeUrl):
            # Both are permanent: the record does not exist, or the URL
            # is refused. Retrying either only spends the budget.
            raise
        except (requests.RequestException, ValueError, TooManyRedirects) as exc:
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
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    validate: bool = True,
) -> ByteResponse:
    """GET ``url``, streaming and capping the body at ``max_bytes``.

    Streamed rather than buffered so an unexpectedly large or endless
    response cannot exhaust memory.

    Every redirect hop is validated. This is the path the crawler takes,
    so a first-party site that redirects to an internal address would
    otherwise be followed there without the guard ever being consulted.
    """
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            request_tracker.record()

            with get_following_redirects(
                url,
                timeout=timeout,
                max_redirects=max_redirects,
                headers=headers or {"Accept": "*/*"},
                stream=True,
                validate=validate,
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

        except UnsafeUrl:
            # Permanent: the address is refused, and retrying cannot
            # change that.
            raise
        except (requests.RequestException, ValueError, TooManyRedirects) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(RETRY_BACKOFF_SECONDS)

    assert last_error is not None
    raise last_error
