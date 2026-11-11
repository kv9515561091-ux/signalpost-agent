"""NAV Arbeidsplassen job-ad connector.

Arbeidsplassen is the national job board run by NAV. A posting names the
employer exactly as registered, carries a publication date and an expiry
date, and has a stable public URL, so it is a dated, checkable statement
that a named company is hiring in a named municipality. That is the
cheapest independent evidence of workforce activity available in Norway:
one national index, no per-company request.

Two properties of the search endpoint shape the design.

The index is Elasticsearch-backed and paginates with ``from``/``size``,
but the result window is hard-capped at 10,000. The national index is
larger than that, so a full harvest partitions by county; every
partition is far below the window.

The endpoint rate-limits aggressively -- a handful of rapid pages earns
a 429 that outlasts a minute. Requests are therefore paced, a 429 is
retried with backoff instead of being treated as a failure, and a
partial harvest is kept rather than discarded.

The harvest is a snapshot, like the Brreg bulk export: refreshing it is
a separate paced pass, and a scored run reads the frozen file so the
2,000-request ceiling is spent on companies rather than on re-listing
the national vacancy index every time.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.request_tracker import request_tracker
from src.text import fold

NAV_SEARCH_URL = "https://arbeidsplassen.nav.no/stillinger/api/search"
AD_URL = "https://arbeidsplassen.nav.no/stillinger/stilling/{uuid}"

DEFAULT_NAV_PATH = Path("data/nav_jobs.json.gz")
DEFAULT_PAGE_SIZE = 100
DEFAULT_MAX_PAGES = 150
# The API rejects ``from`` at 10,000 and above.
RESULT_WINDOW = 10_000
DEFAULT_MIN_INTERVAL = float(os.getenv("NAV_MIN_INTERVAL", "1.0"))
DEFAULT_ATTEMPTS = int(os.getenv("NAV_ATTEMPTS", "4"))
BACKOFF_SECONDS = float(os.getenv("NAV_BACKOFF", "30"))
DEFAULT_MAX_AGE_HOURS = float(os.getenv("NAV_MAX_AGE_HOURS", "12"))

USER_AGENT = "SignalpostAgent/1.0 (+company-intelligence; contact via registry)"

# The 2024 county structure. The filter is best-effort: if the endpoint
# ignores it the harvest simply sees duplicates, which are deduplicated
# by ad uuid, so the failure mode is a smaller index, not a wrong one.
COUNTIES = (
    "OSLO",
    "ROGALAND",
    "MORE OG ROMSDAL",
    "NORDLAND",
    "OSTFOLD",
    "AKERSHUS",
    "BUSKERUD",
    "INNLANDET",
    "VESTFOLD",
    "TELEMARK",
    "AGDER",
    "VESTLAND",
    "TRONDELAG",
    "TROMS",
    "FINNMARK",
)

# A single company rarely needs more than a handful of postings to prove
# it is hiring, and every extra one is another claim in the payload.
MAX_POSTINGS_PER_COMPANY = 10


@dataclass(frozen=True)
class NavPosting:
    """One advertised vacancy, reduced to what a profile can cite."""

    uuid: str
    title: str
    employer: str
    municipality: str | None = None
    county: str | None = None
    published: str | None = None
    expires: str | None = None

    @property
    def url(self) -> str:
        return AD_URL.format(uuid=self.uuid)

    def as_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "title": self.title,
            "employer": self.employer,
            "municipality": self.municipality,
            "county": self.county,
            "published": self.published,
            "expires": self.expires,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NavPosting:
        return cls(
            uuid=str(raw.get("uuid") or ""),
            title=str(raw.get("title") or ""),
            employer=str(raw.get("employer") or ""),
            municipality=raw.get("municipality") or None,
            county=raw.get("county") or None,
            published=raw.get("published") or None,
            expires=raw.get("expires") or None,
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(value: str | None) -> str:
    return fold(value)


def _municipalities(locations: Iterable[dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for location in locations:
        for value in (
            location.get("municipal"),
            location.get("municipality"),
        ):
            key = _key(value)
            if key:
                found.add(key)
    return found


def _posting_from_hit(hit: dict[str, Any]) -> NavPosting | None:
    source = hit.get("_source") if isinstance(hit, dict) else None
    if not isinstance(source, dict):
        return None
    uuid = str(source.get("uuid") or hit.get("_id") or "").strip()
    title = str(source.get("title") or "").strip()
    employer = str(
        (source.get("employer") or {}).get("name")
        or source.get("businessName")
        or ""
    ).strip()
    if not uuid or not title or not employer:
        return None
    locations = [
        location
        for location in source.get("locationList") or []
        if isinstance(location, dict)
    ]
    return NavPosting(
        uuid=uuid,
        title=title,
        employer=employer,
        municipality=(
            str(locations[0].get("municipal") or "").strip() or None
            if locations else None
        ),
        county=(
            str(locations[0].get("county") or "").strip() or None
            if locations else None
        ),
        published=source.get("published") or None,
        expires=source.get("expires") or None,
    )


@dataclass
class NavJobsIndex:
    """A harvested vacancy index, queried by registered employer name."""

    postings: list[NavPosting] = field(default_factory=list)
    retrieved_at: str = ""
    pages_read: int = 0
    total: int = 0
    partitions: list[str] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        self._by_employer: dict[str, list[NavPosting]] = {}
        for posting in self.postings:
            self._by_employer.setdefault(_key(posting.employer), []).append(
                posting
            )

    @property
    def available(self) -> bool:
        return bool(self.postings) and self.error is None

    def employer_keys(self) -> set[str]:
        return set(self._by_employer)

    def employer_name_count(self, name: str | None) -> int:
        """Distinct employer spellings sharing one folded name.

        Folding already merges casing and punctuation variants, so a
        count above one means two genuinely different employer strings
        -- the case where a municipality tie-break is needed.
        """
        key = _key(name)
        if not key:
            return 0
        return len({posting.employer for posting in self._by_employer.get(key, [])})

    def postings_for(self, name: str | None) -> list[NavPosting]:
        return list(self._by_employer.get(_key(name), []))

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieved_at": self.retrieved_at,
            "source": NAV_SEARCH_URL,
            "pages_read": self.pages_read,
            "total": self.total,
            "partitions": self.partitions,
            "truncated": self.truncated,
            "error": self.error,
            "postings": [posting.as_dict() for posting in self.postings],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NavJobsIndex:
        postings = [
            NavPosting.from_dict(item)
            for item in raw.get("postings") or []
            if isinstance(item, dict)
        ]
        return cls(
            postings=[posting for posting in postings if posting.uuid],
            retrieved_at=str(raw.get("retrieved_at") or ""),
            pages_read=int(raw.get("pages_read") or 0),
            total=int(raw.get("total") or 0),
            partitions=list(raw.get("partitions") or []),
            truncated=bool(raw.get("truncated")),
            error=raw.get("error") or None,
        )


def match_postings(
    index: NavJobsIndex,
    *,
    company_name: str | None,
    previous_names: Iterable[str] = (),
    subunit_names: Iterable[str] = (),
    municipalities: Iterable[str] = (),
) -> list[NavPosting]:
    """Return postings attributable to one company, or none.

    Attribution is exact-name only. A posting is accepted when its
    employer name equals the registered legal name (or a previous name)
    and that name is unique in the index, or when the name is shared and
    at least one posting municipality equals one the company has on
    record. Registered workplace names are matched the same way, which is
    what catches "KONGSBERG MARITIME ASA AVD ..." ads. Anything ambiguous
    is dropped: a wrong-company job ad voids the submission.
    """
    wanted = {_key(value) for value in municipalities if _key(value)}
    accepted: dict[str, NavPosting] = {}

    names = [company_name, *previous_names]
    for name in names:
        key = _key(name)
        if not key or key not in index.employer_keys():
            continue
        postings = index.postings_for(name)
        if index.employer_name_count(name) == 1:
            for posting in postings:
                accepted.setdefault(posting.uuid, posting)
            continue
        for posting in postings:
            if _key(posting.municipality) in wanted:
                accepted.setdefault(posting.uuid, posting)

    for name in subunit_names:
        key = _key(name)
        if not key or key not in index.employer_keys():
            continue
        postings = index.postings_for(name)
        if index.employer_name_count(name) == 1:
            for posting in postings:
                accepted.setdefault(posting.uuid, posting)
            continue
        for posting in postings:
            if _key(posting.municipality) in wanted:
                accepted.setdefault(posting.uuid, posting)

    return sorted(
        accepted.values(),
        key=lambda posting: (posting.published or "", posting.uuid),
        reverse=True,
    )


def job_facts(
    postings: list[NavPosting],
    *,
    retrieved_at: str,
) -> list[dict[str, Any]]:
    """Fact records for a company's advertised vacancies.

    One claim per posting, plus a count, so a reviewer sees the
    individual ad rather than only an aggregate.
    """
    if not postings:
        return []

    facts: list[dict[str, Any]] = []
    for posting in postings[:MAX_POSTINGS_PER_COMPANY]:
        where = f" in {posting.municipality}" if posting.municipality else ""
        published = (posting.published or "")[:10]
        expiry = (posting.expires or "")[:10]
        detail = f", published {published}" if published else ""
        if expiry:
            detail += f", expires {expiry}"
        facts.append(
            {
                "field": "job_opening",
                "value": posting.title,
                "source_url": posting.url,
                "retrieved_at": retrieved_at,
                "evidence_text": (
                    f"NAV Arbeidsplassen advertises \"{posting.title}\" "
                    f"at {posting.employer}{where}{detail}"
                ),
                "confidence": 0.7,
                "source_tier": 4,
                "source_type": "official_job_feed",
                "as_of": (posting.published or "")[:10] or None,
            }
        )

    facts.append(
        {
            "field": "jobs_count",
            "value": len(postings),
            "source_url": postings[0].url,
            "retrieved_at": retrieved_at,
            "evidence_text": (
                f"{len(postings)} active vacancies listed on NAV "
                f"Arbeidsplassen under this employer name"
            ),
            "confidence": 0.7,
            "source_tier": 4,
            "source_type": "official_job_feed",
            "as_of": (postings[0].published or "")[:10] or None,
        }
    )
    return facts


def _request_page(
    params: dict[str, Any],
    *,
    timeout: int = 30,
) -> dict[str, Any]:
    request_tracker.record()
    response = requests.get(
        NAV_SEARCH_URL,
        params=params,
        timeout=timeout,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    if response.status_code == 429:
        raise _RateLimited(response.headers.get("Retry-After"))
    response.raise_for_status()
    return response.json()


class _RateLimited(Exception):
    def __init__(self, retry_after: str | None = None):
        super().__init__("rate limited")
        self.retry_after = retry_after


def _fetch_with_retry(
    fetch: Callable[[dict[str, Any]], dict[str, Any]],
    params: dict[str, Any],
    *,
    attempts: int,
    backoff: float,
    sleep: Callable[[float], None],
) -> dict[str, Any] | None:
    for attempt in range(attempts):
        try:
            return fetch(params)
        except _RateLimited as exc:
            wait = backoff
            if exc.retry_after:
                with contextlib.suppress(ValueError):
                    wait = max(wait, float(exc.retry_after))
            sleep(min(wait, 300))
        except Exception:  # noqa: BLE001 - one page must not end the harvest
            if attempt + 1 >= attempts:
                return None
            sleep(backoff)
    return None


def _scope_total(payload: dict[str, Any]) -> int:
    total = (payload.get("hits") or {}).get("total") or {}
    if isinstance(total, dict):
        return int(total.get("value") or 0)
    return int(total or 0)


def _scope_hits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list((payload.get("hits") or {}).get("hits") or [])


def harvest(
    *,
    fetch: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int = DEFAULT_MAX_PAGES,
    min_interval: float = DEFAULT_MIN_INTERVAL,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[int, int, int], None] | None = None,
    window: int = RESULT_WINDOW,
) -> NavJobsIndex:
    """Read the national vacancy index, partitioned when it is too large.

    ``fetch`` is injectable so the pagination and partition logic can be
    tested without touching the live endpoint. ``window`` is the
    endpoint's own pagination ceiling; it is a parameter only so tests
    can shrink it.
    """
    fetch = fetch or _request_page
    postings: dict[str, NavPosting] = {}
    pages = 0
    partitions: list[str] = []
    truncated = False
    errors: list[str] = []

    def read(params: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal pages
        result = _fetch_with_retry(
            fetch, params, attempts=attempts, backoff=backoff, sleep=sleep
        )
        if result is not None:
            pages += 1
        return result

    probe = read({"size": page_size, "from": 0})
    if probe is None:
        return NavJobsIndex(
            retrieved_at=_now(), error="NAV search endpoint unreachable"
        )

    total = _scope_total(probe)
    if total > window:
        scopes: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = [
            ("county:" + county, {"county": county}, None)
            for county in COUNTIES
        ]
    else:
        scopes = [("national", {}, probe)]

    seen_uuids: set[str] = set()
    queue = list(scopes)
    position = 0
    while position < len(queue):
        label, extra, first_payload = queue[position]
        position += 1
        if pages >= max_pages:
            truncated = True
            break

        payload = first_payload or read({"size": page_size, "from": 0, **extra})
        if payload is None:
            errors.append(f"{label}: unreachable")
            continue

        scope_total = _scope_total(payload)
        if (
            extra
            and scope_total == total
            and total > window
        ):
            # The endpoint ignored the county filter. Every partition
            # would replay the same list, so fall back to the single
            # window the API allows instead of spending the budget on
            # fifteen copies of it.
            errors.append("county filter ignored; harvested first window")
            queue = [("national", {}, probe)]
            position = 0
            postings.clear()
            seen_uuids.clear()
            partitions.clear()
            continue

        partitions.append(label)
        offset = 0
        while True:
            hits = _scope_hits(payload)
            if not hits:
                break
            for hit in hits:
                posting = _posting_from_hit(hit)
                if posting is not None and posting.uuid not in seen_uuids:
                    seen_uuids.add(posting.uuid)
                    postings[posting.uuid] = posting
            offset += page_size
            if progress is not None:
                progress(pages, max_pages, len(postings))
            if offset >= min(scope_total, window):
                break
            if len(hits) < page_size:
                break
            if pages >= max_pages:
                truncated = True
                break
            sleep(min_interval)
            payload = read({"size": page_size, "from": offset, **extra})
            if payload is None:
                errors.append(f"{label}@{offset}: unreachable")
                break

    index = NavJobsIndex(
        postings=list(postings.values()),
        retrieved_at=_now(),
        pages_read=pages,
        total=total,
        partitions=partitions,
        truncated=truncated,
        error="; ".join(errors) if errors else None,
    )
    return index


def save(index: NavJobsIndex, path: str | Path = DEFAULT_NAV_PATH) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(index.as_dict(), handle, ensure_ascii=False)
    os.replace(temporary, destination)
    return destination


def load(path: str | Path = DEFAULT_NAV_PATH) -> NavJobsIndex | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return NavJobsIndex.from_dict(raw)


def _age_hours(retrieved_at: str) -> float | None:
    try:
        stamp = datetime.fromisoformat(retrieved_at)
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - stamp).total_seconds() / 3600


def ensure_snapshot(
    path: str | Path = DEFAULT_NAV_PATH,
    *,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    refresh: bool = False,
    **harvest_options: Any,
) -> tuple[NavJobsIndex | None, dict[str, Any]]:
    """Return a usable vacancy index, refreshing a stale snapshot.

    A refresh that fails keeps the previous snapshot rather than
    dropping the family: a day-old vacancy list is still dated evidence,
    and the metadata records that it is old.
    """
    existing = load(path)
    age = _age_hours(existing.retrieved_at) if existing else None
    fresh = existing is not None and age is not None and age <= max_age_hours

    if existing is not None and fresh and not refresh:
        return existing, {
            "path": str(path),
            "retrieved_at": existing.retrieved_at,
            "age_hours": round(age or 0.0, 1),
            "postings": len(existing.postings),
            "pages_read": existing.pages_read,
            "total": existing.total,
            "stale": False,
            "refreshed": False,
        }

    harvested = harvest(**harvest_options)
    if harvested.postings:
        save(harvested, path)
        return harvested, {
            "path": str(path),
            "retrieved_at": harvested.retrieved_at,
            "age_hours": 0.0,
            "postings": len(harvested.postings),
            "pages_read": harvested.pages_read,
            "total": harvested.total,
            "partitions": harvested.partitions,
            "truncated": harvested.truncated,
            "stale": False,
            "refreshed": True,
        }

    if existing is not None:
        return existing, {
            "path": str(path),
            "retrieved_at": existing.retrieved_at,
            "age_hours": round(age or 0.0, 1),
            "postings": len(existing.postings),
            "pages_read": existing.pages_read,
            "total": existing.total,
            "stale": True,
            "refreshed": False,
            "error": harvested.error or "refresh produced no postings",
        }

    return None, {
        "path": str(path),
        "error": harvested.error or "refresh produced no postings",
        "refreshed": False,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.nav_jobs",
        description="Harvest the NAV Arbeidsplassen vacancy index.",
    )
    parser.add_argument("--path", default=str(DEFAULT_NAV_PATH))
    parser.add_argument("--pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--interval", type=float, default=DEFAULT_MIN_INTERVAL)
    args = parser.parse_args(argv)

    def progress(pages: int, cap: int, postings: int) -> None:
        print(f"  page {pages}/{cap}: {postings} postings", flush=True)

    index = harvest(
        max_pages=args.pages,
        min_interval=args.interval,
        progress=progress,
    )
    if index.postings:
        save(index, args.path)
    print(
        f"  total advertised {index.total}, collected "
        f"{len(index.postings)} in {index.pages_read} pages"
    )
    if index.error:
        print(f"  partial: {index.error}")
    return 0 if index.postings else 1


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
