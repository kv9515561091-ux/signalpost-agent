"""Which companies produced facts last time.

The daily limit that binds is wall clock, not requests: a company with
no extractable facts costs exactly as much as one with thirty, because
both walk their sources and pay a model call per page. Remembering the
outcome lets the next run spend that time where it earns something and
give the rest a shallower read, at the end.

Two deliberate choices:

* An empty company is still read. Coverage is scored per company, and
  yesterday's empty page can be today's update, so this is a priority
  hint, never a blacklist.
* The cache is only written when a company was actually judged. A
  company skipped for the request or time budget has no outcome, and
  recording a zero for it would teach the next run a false lesson.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("data/yield_cache.json")

# A company that produced nothing last time still gets read; it just
# does not get the full walk. The best sources are tried first, so a
# shallower ceiling drops the tail of the queue, not the head.
ZERO_YIELD_PAGES = 6

# Bands, lowest first. A stable sort by this keeps the caller's input
# order inside each band, so the output file is unaffected.
PRIORITY_PRODUCTIVE = 0
PRIORITY_UNKNOWN = 1
PRIORITY_EMPTY = 2

_lock = threading.RLock()
_entries: dict[str, dict[str, Any]] | None = None
_configured: Path | None = None
_default: Path | None = None


def enabled() -> bool:
    raw = (os.getenv("AGENT_YIELD_CACHE") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def use_path(path: str | Path) -> None:
    """Point the cache at a specific file, overriding the run default."""
    global _configured, _entries
    with _lock:
        resolved = Path(path)
        if resolved != _configured:
            _configured = resolved
            _entries = None


def use_default(path: str | Path) -> None:
    """Say where the cache belongs for this run.

    The cache sits beside the output it describes, so two runs writing
    different files cannot overwrite each other's history. An explicit
    ``use_path`` or ``AGENT_YIELD_CACHE_PATH`` still wins.
    """
    global _default, _entries
    with _lock:
        resolved = Path(path)
        if resolved != _default:
            _default = resolved
            _entries = None


def _path() -> Path:
    env = os.getenv("AGENT_YIELD_CACHE_PATH")
    if env:
        return Path(env)
    if _configured is not None:
        return _configured
    if _default is not None:
        return _default
    return DEFAULT_PATH


def load() -> dict[str, dict[str, Any]]:
    """The whole cache, read once per process."""
    global _entries
    with _lock:
        if _entries is None:
            try:
                loaded = json.loads(_path().read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = {}
            _entries = loaded if isinstance(loaded, dict) else {}
        return _entries


def reset() -> None:
    """Drop the in-process copy. Tests use it; a run never needs to."""
    global _entries
    with _lock:
        _entries = None


def facts_last_time(company_number: str) -> int | None:
    """Facts this company produced last run, or ``None`` if unknown."""
    if not enabled():
        return None
    entry = load().get(company_number)
    if not isinstance(entry, dict):
        return None
    try:
        return int(entry.get("facts", 0))
    except (TypeError, ValueError):
        return None


def priority(company_number: str) -> int:
    """Ordering band: productive first, unknown next, empty last."""
    if not enabled():
        return PRIORITY_UNKNOWN
    known = facts_last_time(company_number)
    if known is None:
        return PRIORITY_UNKNOWN
    return PRIORITY_PRODUCTIVE if known > 0 else PRIORITY_EMPTY


def domains_last_time(company_number: str) -> list[str]:
    """Hosts that produced at least one fact last run."""
    entry = load().get(company_number)
    if not isinstance(entry, dict):
        return []
    domains = entry.get("domains")
    if not isinstance(domains, list):
        return []
    return [str(domain) for domain in domains if domain]


def record(
    company_number: str,
    *,
    facts: int,
    domains: list[str] | set[str] | tuple[str, ...] = (),
) -> None:
    """Store this company's outcome. Never raises."""
    if not enabled():
        return

    with _lock:
        entries = load()
        entries[company_number] = {
            "facts": int(facts),
            "domains": sorted({str(domain) for domain in domains if domain}),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        _save(entries)


def _save(entries: dict[str, dict[str, Any]]) -> None:
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(entries, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError:
        # A cache that cannot be written must not break a run.
        pass


__all__ = [
    "DEFAULT_PATH",
    "PRIORITY_EMPTY",
    "PRIORITY_PRODUCTIVE",
    "PRIORITY_UNKNOWN",
    "ZERO_YIELD_PAGES",
    "domains_last_time",
    "enabled",
    "facts_last_time",
    "load",
    "priority",
    "record",
    "reset",
    "use_path",
]
