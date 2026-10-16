"""
Change detection between two profiles.

The daily test re-runs the same hundred companies. "Update correctly" is
worth twenty points, and until now nothing in the agent could answer the
only question that category asks: *what changed since yesterday?*

The registry will not help. Enhetsregisteret exposes no change signal on
an entity -- no ``endrettsdato``, no ETag, no Last-Modified, and a
``Cache-Control: no-cache`` that says it does not track them either. So
the comparison has to be made against the previous profile rather than
inferred from the source.

Two kinds of change are reported separately, because they carry
different weight:

* a **field** changed -- the registered address, a director, a filed
  figure. Authoritative, because the registry is the spine.
* a **fact** changed -- something extracted from the web. Weaker, and
  reported as such, because it may reflect a page being rewritten
  rather than the company changing.

Every change carries when it was observed and what it was before. A
change record without the previous value is only half an audit trail, and
"something changed" is not actionable.
"""

from __future__ import annotations

from typing import Any

# Fields worth comparing. Presence and absence both matter: a field that
# appeared or disappeared is more often a real change than a field that
# merely held steady.
TRACKED_FIELDS = (
    "company_name",
    "previous_names",
    "legal_form",
    "status",
    "industry",
    "industry_code",
    "business_description",
    "founding_date",
    "registration_date",
    "last_filed_annual_accounts",
    "address",
    "business_address",
    "postal_address",
    "municipality",
    "employee_count",
    "website",
    "phone",
    "email",
    "bankruptcy",
    "under_liquidation",
    "is_group_parent",
    "parent_company_number",
    "ceo_name",
    "board_chair_name",
    "board_size",
    "auditor_name",
    "owner_name",
    "financial_currency",
)


def _key(value: Any) -> Any:
    """A comparable form, so 'None' and 'not present' are the same."""
    if isinstance(value, (list, tuple)):
        return tuple(sorted(str(item) for item in value))
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _normalise_fact(fact: dict[str, Any]) -> tuple[str, str]:
    return (
        _key(fact.get("field")) or "",
        _key(fact.get("value")) or "",
    )


def diff_profiles(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[dict[str, Any]]:
    """What changed between two versions of one company.

    Args:
        previous: The stored profile from an earlier run, or ``None`` on
            a first sighting -- which is itself reportable, so a caller
            can distinguish "first time seen" from "no change".
        current: The profile just built.

    Returns:
        One record per change, each with ``field``, ``previous``,
        ``current`` and ``kind``. Empty when nothing moved.
    """
    if previous is None:
        return [
            {
                "field": "__profile__",
                "previous": None,
                "current": current.get("company_number"),
                "kind": "first_seen",
                "note": "no previous profile was stored for this company",
            }
        ]

    changes: list[dict[str, Any]] = []

    for name in TRACKED_FIELDS:
        if name not in current and name not in previous:
            continue

        before = previous.get(name)
        after = current.get(name)

        if _key(before) != _key(after):
            changes.append(
                {
                    "field": name,
                    "previous": before,
                    "current": after,
                    "kind": _kind_of(name),
                }
            )

    changes.extend(_diff_roles(previous, current))
    changes.extend(_diff_financials(previous, current))
    changes.extend(_diff_facts(previous, current))

    return changes


# Which body of evidence a field comes from, so a reader knows how much
# to trust the change.
_REGISTRY_FIELDS = frozenset(
    {
        "company_name", "previous_names", "legal_form", "status",
        "industry", "industry_code", "business_description",
        "founding_date", "registration_date", "last_filed_annual_accounts",
        "address", "business_address", "postal_address", "municipality",
        "employee_count", "bankruptcy", "under_liquidation",
        "is_group_parent", "parent_company_number", "financial_currency",
    }
)


def _kind_of(field: str) -> str:
    if field in _REGISTRY_FIELDS:
        return "registry"
    if field in ("website", "phone", "email"):
        return "contact"
    return "management"


def _diff_roles(
    previous: dict[str, Any], current: dict[str, Any]
) -> list[dict[str, Any]]:
    """Board and management changes, keyed by role rather than by name."""
    before = _roles_by_name(previous)
    after = _roles_by_name(current)

    changes: list[dict[str, Any]] = []

    for name, role in after.items():
        if name not in before:
            changes.append(
                {
                    "field": f"roles.{role}",
                    "previous": None,
                    "current": name,
                    "kind": "registry",
                    "note": "role newly held",
                }
            )
        elif before[name] != role:
            changes.append(
                {
                    "field": f"roles.{role}",
                    "previous": before[name],
                    "current": role,
                    "kind": "registry",
                    "note": "role changed",
                }
            )

    for name, role in before.items():
        if name not in after:
            changes.append(
                {
                    "field": f"roles.{role}",
                    "previous": name,
                    "current": None,
                    "kind": "registry",
                    "note": "role no longer held",
                }
            )

    return changes


def _roles_by_name(profile: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for record in profile.get("roles") or []:
        if not isinstance(record, dict):
            continue
        name = str(record.get("name") or "").strip()
        role = str(record.get("role_label") or record.get("role") or "").strip()
        if name and role:
            out[name] = role
    return out


def _diff_financials(
    previous: dict[str, Any], current: dict[str, Any]
) -> list[dict[str, Any]]:
    """Filed figures, compared per fiscal year.

    Keyed by year rather than by position so that a newly filed year
    reads as an addition and not as every later year having changed.
    """
    before = _financials_by_year(previous)
    after = _financials_by_year(current)

    changes: list[dict[str, Any]] = []

    for year, figures in sorted(after.items()):
        if year not in before:
            changes.append(
                {
                    "field": f"financials.{year}",
                    "previous": None,
                    "current": figures,
                    "kind": "registry",
                    "note": "newly filed annual accounts",
                }
            )
            continue

        for measure, value in sorted(figures.items()):
            if _key(before[year].get(measure)) != _key(value):
                changes.append(
                    {
                        "field": f"financials.{year}.{measure}",
                        "previous": before[year].get(measure),
                        "current": value,
                        "kind": "registry",
                    }
                )

    for year in sorted(set(before) - set(after)):
        changes.append(
            {
                "field": f"financials.{year}",
                "previous": before[year],
                "current": None,
                "kind": "registry",
                "note": "annual accounts no longer present",
            }
        )

    return changes


_FIGURES = (
    "revenue", "operating_profit", "profit_before_tax", "net_profit",
    "total_assets", "total_liabilities", "equity", "operating_costs",
)


def _financials_by_year(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in profile.get("financials") or []:
        if not isinstance(record, dict):
            continue
        year = str(record.get("fiscal_year") or "")
        if not year:
            continue
        out[year] = {m: record.get(m) for m in _FIGURES}
    return out


def _diff_facts(
    previous: dict[str, Any], current: dict[str, Any]
) -> list[dict[str, Any]]:
    """Extracted facts, compared by content.

    Reported separately from registry changes because their provenance is
    a web page rather than the register: a changed fact may mean the
    company changed, or that a page was rewritten.
    """
    before = {_normalise_fact(f): f for f in previous.get("facts") or [] if isinstance(f, dict)}
    after = {_normalise_fact(f): f for f in current.get("facts") or [] if isinstance(f, dict)}

    changes: list[dict[str, Any]] = []

    for key, fact in after.items():
        if key not in before:
            changes.append(
                {
                    "field": f"fact.{fact.get('field')}",
                    "previous": None,
                    "current": fact.get("value"),
                    "kind": "extracted",
                    "source_url": fact.get("source_url"),
                }
            )
        else:
            old = before[key]
            if _key(old.get("evidence_text")) != _key(fact.get("evidence_text")):
                changes.append(
                    {
                        "field": f"fact.{fact.get('field')}",
                        "previous": old.get("evidence_text"),
                        "current": fact.get("evidence_text"),
                        "kind": "extracted_rewording",
                        "source_url": fact.get("source_url"),
                        "note": "same value, different quoted evidence",
                    }
                )

    for key, fact in before.items():
        if key not in after:
            changes.append(
                {
                    "field": f"fact.{fact.get('field')}",
                    "previous": fact.get("value"),
                    "current": None,
                    "kind": "extracted",
                    "source_url": fact.get("source_url"),
                    "note": "fact no longer found",
                }
            )

    return changes


def summarise_changes(changes: list[dict[str, Any]]) -> dict[str, int]:
    """Counts by kind, for a one-line report."""
    counts: dict[str, int] = {}
    for change in changes:
        kind = str(change.get("kind") or "other")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


__all__ = [
    "TRACKED_FIELDS",
    "diff_profiles",
    "summarise_changes",
]
