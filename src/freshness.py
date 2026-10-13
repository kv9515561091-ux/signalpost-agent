"""
Freshness metadata.

Two distinct clocks matter and they must not be conflated:

``retrieved_at``
    When *the agent* fetched the fact. This is nearly always minutes ago,
    so it carries almost no information about whether the fact itself is
    current. Measuring "freshness" purely from it marks everything fresh
    forever, which is how a three-year-old figure ends up presented as
    current.

``as_of`` / ``reporting_period``
    When the *fact* describes. This is the real freshness signal: a
    revenue figure for fiscal year 2023 is current as of 2023-12-31
    regardless of when the agent read it.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

# How stale a fact's subject matter may be before it is flagged, by the
# kind of thing being described.
FRESHNESS_WINDOW_DAYS = {
    "live_registry": None,
    "registry": 120,
    "financial": 500,
    "news": 180,
    "web": 90,
}

STATUS_CURRENT = "current"
STATUS_AGING = "aging"
STATUS_STALE = "stale"
STATUS_UNKNOWN = "unknown"


def _parse(value: str | None) -> date | None:
    """Parse the ISO-ish dates Norwegian registries actually emit."""
    if not value:
        return None

    text = str(value).strip()

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d.%m.%Y", "%Y-%m"):
        try:
            parsed = datetime.strptime(text[: len(fmt) + 6], fmt)
        except ValueError:
            continue
        return parsed.date()

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def age_in_days(as_of: str | None, today: date | None = None) -> float | None:
    """Days between ``as_of`` and today. ``None`` when unknown."""
    parsed = _parse(as_of)
    if parsed is None:
        return None

    reference = today or datetime.now(timezone.utc).date()
    return max(0.0, (reference - parsed).days)


def assess(as_of: str | None, kind: str = "registry") -> dict:
    """Classify how current a fact's subject matter is.

    Args:
        as_of: The date the fact describes — a reporting period end,
            a publication date, or a registry change date. ``None`` for
            facts that are read live from a continuously maintained
            registry, where retrieval time is itself the guarantee.
        kind: Key into :data:`FRESHNESS_WINDOW_DAYS`.
    """
    window = FRESHNESS_WINDOW_DAYS.get(kind, FRESHNESS_WINDOW_DAYS["registry"])

    # A live registry read asserts the value as of the moment of reading.
    # There is no publication lag to measure, so it is current by
    # definition rather than unknown.
    if window is None:
        return {
            "as_of": as_of,
            "age_days": 0.0,
            "window_days": None,
            "status": STATUS_CURRENT,
        }

    age = age_in_days(as_of)

    if age is None:
        status = STATUS_UNKNOWN
    elif age <= window:
        status = STATUS_CURRENT
    elif age <= window * 2:
        status = STATUS_AGING
    else:
        status = STATUS_STALE

    return {
        "as_of": as_of,
        "age_days": round(age, 0) if age is not None else None,
        "window_days": window,
        "status": status,
    }


def status_of(as_of: str | None, kind: str = "registry") -> str:
    """Shorthand for ``assess(...)["status"]``."""
    return assess(as_of, kind)["status"]
