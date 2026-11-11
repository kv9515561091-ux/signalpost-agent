"""Dated activity read from a verified first-party page.

A company site's news block is the only "what happened recently" signal
available without a search provider or a model: a headline next to a
date, on a page whose identity has already been proven. The claim is
narrow on purpose -- the page lists this item under this date -- because
that is exactly what the page states.

Markup is already reduced to text by ``src.page``, so this works on
lines. A date is accepted only when a plausible headline sits on the
same line or immediately beside it, the date is in the past, and it is
recent enough to count as activity. Anything else is dropped rather
than guessed.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

MONTHS = {
    "januar": 1,
    "februar": 2,
    "mars": 3,
    "april": 4,
    "mai": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "desember": 12,
}

_ABBREVIATIONS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "okt": 10,
    "nov": 11,
    "des": 12,
}

_NUMERIC = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?!\d)")
_WORDED = re.compile(
    r"\b(\d{1,2})\.?\s+(" + "|".join([*MONTHS, *_ABBREVIATIONS]) + r")\s+(\d{4})\b",
    re.IGNORECASE,
)

# Lines that carry a date but are never company activity.
_NOISE = (
    "cookie",
    "personvern",
    "personvernerklæring",
    "alle rettigheter",
    "©",
    "(c)",
    "logg inn",
    "log in",
    "org.nr",
    "organisasjonsnummer",
    "mva",
)

MIN_TITLE = 15
MAX_TITLE = 180
MAX_AGE_DAYS = 730
MAX_ITEMS = 3


def _to_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def find_date(text: str) -> tuple[date, tuple[int, int]] | None:
    """Return the first date in ``text`` and the span it occupied."""
    for pattern, build in (
        (_ISO, lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        (_NUMERIC, lambda m: (int(m.group(3)), int(m.group(2)), int(m.group(1)))),
        (
            _WORDED,
            lambda m: (
                int(m.group(3)),
                MONTHS.get(m.group(2).lower())
                or _ABBREVIATIONS.get(m.group(2).lower()),
                int(m.group(1)),
            ),
        ),
    ):
        match = pattern.search(text)
        if not match:
            continue
        parts = build(match)
        if any(part is None for part in parts):
            continue
        found = _to_date(*parts)  # type: ignore[arg-type]
        if found is not None:
            return found, (match.start(), match.end())
    return None


def _clean(value: str) -> str:
    return " ".join(value.split()).strip(" -–—:|•·\t")


def _usable(title: str) -> bool:
    if not (MIN_TITLE <= len(title) <= MAX_TITLE):
        return False
    if sum(ch.isalpha() for ch in title) < 10:
        return False
    lowered = title.casefold()
    return not any(marker in lowered for marker in _NOISE)


def _title_candidates(lines: list[str], index: int, span: tuple[int, int]):
    line = lines[index]
    yield _clean(line[: span[0]] + " " + line[span[1] :])
    for step in (-1, 1):
        position = index + step
        while 0 <= position < len(lines) and not lines[position]:
            position += step
        if 0 <= position < len(lines):
            yield _clean(lines[position])


def dated_items(
    text: str,
    *,
    today: date | None = None,
    max_age_days: int = MAX_AGE_DAYS,
    limit: int = MAX_ITEMS,
) -> list[dict]:
    """Dated headlines from a page's text, newest first.

    Returns ``{"title", "date", "evidence_text"}`` for each accepted
    item. Duplicate headlines are collapsed; the same story linked twice
    on a page is one activity, not two.
    """
    if not text:
        return []

    reference = today or datetime.now().date()
    oldest = reference - timedelta(days=max_age_days)
    newest = reference + timedelta(days=1)

    lines = [line.strip() for line in text.splitlines()]
    items: list[dict] = []
    seen: set[str] = set()

    for index, line in enumerate(lines):
        if not line:
            continue
        found = find_date(line)
        if found is None:
            continue
        when, span = found
        if not (oldest <= when <= newest):
            continue
        for candidate in _title_candidates(lines, index, span):
            if not _usable(candidate):
                continue
            key = candidate.casefold()
            if key in seen:
                break
            seen.add(key)
            items.append(
                {
                    "title": candidate,
                    "date": when.isoformat(),
                    "evidence_text": _clean(line),
                }
            )
            break

    items.sort(key=lambda item: item["date"], reverse=True)
    return items[:limit]
