"""
Disk cache for registry and page responses.

Re-running the agent over the same organizations re-fetches everything,
which costs money on the search side and burns the request budget for no
new information. Registry records change rarely, so caching them is safe
within a short window.

Two rules keep the cache honest:

* Only GET responses are cached. A cached body is always keyed by its
  full URL, so two different organizations can never share an entry.
* A stale entry is served only when revalidation fails. If the network is
  down, yesterday's registry record beats no profile at all — provided
  the caller is told, via ``from_stale``, rather than being handed old
  data as though it were fresh.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.http import ByteResponse, fetch_bytes, fetch_json

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_TTL_SECONDS = 6 * 60 * 60  # 6 hours

# Search results go stale faster than registry records: a page's wording
# and ranking change, while a company's legal identity does not. An hour
# still covers a full batch run and any immediate re-run.
SEARCH_TTL_SECONDS = 60 * 60


@dataclass
class CacheEntry:
    """One cached response."""

    url: str
    body: Any
    retrieved_at: str
    stored_at: float
    from_stale: bool = False

    @property
    def age_seconds(self) -> float:
        return time.time() - self.stored_at


class DiskCache:
    """A content-addressed JSON cache on disk."""

    def __init__(
        self,
        directory: str | Path = DEFAULT_CACHE_DIR,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self.directory = Path(directory)
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.stale_serves = 0
        self.writes = 0

    # ---- keys ---------------------------------------------------------
    def _path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.directory / f"{digest}.json"

    def _read(self, url: str) -> dict[str, Any] | None:
        path = self._path(url)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write(self, url: str, body: Any, retrieved_at: str) -> None:
        path = self._path(url)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "url": url,
                        "body": body,
                        "retrieved_at": retrieved_at,
                        "stored_at": time.time(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.writes += 1
        except OSError:
            # A cache that cannot be written must not break the run.
            pass

    # ---- lookups ------------------------------------------------------
    def get_fresh(self, url: str) -> Any | None:
        """Return a cached body still inside its TTL, else ``None``."""
        record = self._read(url)
        if record is None:
            self.misses += 1
            return None

        if time.time() - record.get("stored_at", 0) <= self.ttl_seconds:
            self.hits += 1
            return record["body"]

        return None

    def get_stale(self, url: str) -> Any | None:
        """Return an expired entry, for use when revalidation fails."""
        record = self._read(url)
        return record["body"] if record else None

    def put(self, url: str, body: Any, retrieved_at: str) -> None:
        self._write(url, body, retrieved_at)

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stale_serves": self.stale_serves,
            "writes": self.writes,
        }

    def clear(self) -> int:
        """Delete every entry. Returns how many were removed."""
        if not self.directory.exists():
            return 0
        removed = 0
        for path in self.directory.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed


# A process-wide cache. The registry spine is authoritative within a run,
# so sharing one instance also prevents fetching the same org twice.
cache = DiskCache()

# Caches keyed by (directory, ttl). Constructing a fresh DiskCache per
# call would work but resets the hit/miss counters, so a run using a
# non-default TTL would report nothing.
_caches: dict[tuple[str, int], DiskCache] = {
    (str(cache.directory), cache.ttl_seconds): cache,
}


def cache_for(directory: str | Path, ttl_seconds: int) -> DiskCache:
    """Return the shared cache for a directory and TTL, creating it once."""
    key = (str(directory), ttl_seconds)
    existing = _caches.get(key)
    if existing is None:
        existing = DiskCache(directory, ttl_seconds)
        _caches[key] = existing
    return existing


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def cached_json(
    url: str,
    *,
    accept_missing: bool = False,
    ttl_seconds: int | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """GET ``url`` as JSON, consulting the cache first.

    Args:
        headers: Extra request headers, e.g. API credentials. These are
            sent but never persisted: only the URL reaches disk.

    Returns ``(body, meta)`` where ``meta`` records whether the body came
    from cache, whether it was stale, and when it was originally
    retrieved, so provenance still points at the real source.

    Note:
        The caller must pass a URL that fully identifies the request,
        query parameters included. This function keys on the URL alone,
        so an endpoint that varies results by query string — as the
        search API does — must be given its expanded URL, or every query
        will read the same entry.
    """
    active = cache if ttl_seconds is None else cache_for(
        cache.directory, ttl_seconds
    )

    hit = active.get_fresh(url)
    if hit is not None:
        record = active._read(url) or {}
        return hit, {
            "url": url,
            "retrieved_at": record.get("retrieved_at"),
            "cached": True,
            "stale": False,
        }

    try:
        body = fetch_json(
            url, accept_missing=accept_missing, headers=headers
        )
    except Exception:  # noqa: BLE001 - fall back rather than fail outright
        stale = active.get_stale(url)
        if stale is not None:
            active.stale_serves += 1
            record = active._read(url) or {}
            return stale, {
                "url": url,
                "retrieved_at": record.get("retrieved_at"),
                "cached": True,
                "stale": True,
            }
        raise

    retrieved_at = _now_iso()
    if body is not None:
        active.put(url, body, retrieved_at)

    return body, {
        "url": url,
        "retrieved_at": retrieved_at,
        "cached": False,
        "stale": False,
    }


def cached_page(
    url: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> tuple[ByteResponse, dict[str, Any]]:
    """GET a page body, consulting the cache first.

    Pages are stored base64-encoded because their bodies are arbitrary
    bytes, not JSON.
    """
    import base64

    active = (
        cache
        if ttl_seconds == DEFAULT_TTL_SECONDS
        else cache_for(cache.directory, ttl_seconds)
    )

    hit = active.get_fresh(url)
    if hit is not None:
        record = active._read(url) or {}
        return _decode_page(hit), {
            "url": url,
            "retrieved_at": record.get("retrieved_at"),
            "cached": True,
            "stale": False,
        }

    try:
        response = fetch_bytes(url, max_bytes=2_000_000)
    except Exception:  # noqa: BLE001
        stale = active.get_stale(url)
        if stale is not None:
            active.stale_serves += 1
            record = active._read(url) or {}
            return _decode_page(stale), {
                "url": url,
                "retrieved_at": record.get("retrieved_at"),
                "cached": True,
                "stale": True,
            }
        raise

    retrieved_at = _now_iso()
    active.put(
        url,
        {
            "body_b64": base64.b64encode(response.body).decode("ascii"),
            "encoding": response.encoding,
            "content_type": response.content_type,
            "truncated": response.truncated,
            "status_code": response.status_code,
        },
        retrieved_at,
    )

    return response, {
        "url": url,
        "retrieved_at": retrieved_at,
        "cached": False,
        "stale": False,
    }


def _decode_page(stored: dict[str, Any]) -> ByteResponse:
    import base64

    return ByteResponse(
        body=base64.b64decode(stored["body_b64"]),
        encoding=stored.get("encoding") or "utf-8",
        content_type=stored.get("content_type") or "",
        truncated=bool(stored.get("truncated")),
        status_code=int(stored.get("status_code") or 200),
    )
