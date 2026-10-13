"""
Conflict detection between sources.

When two credible sources disagree about a field, the agent must not
silently pick one. Picking silently converts a known uncertainty into a
confident-looking answer, which is precisely the failure mode the agent
exists to prevent. Conflicting values are therefore surfaced, attributed
to both sources, and resolved only when a rule clearly outranks them —
otherwise both are published and the disagreement is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.text import fold


@dataclass
class FieldValue:
    """One asserted value for one field, with its provenance."""

    field: str
    value: Any
    source_url: str
    retrieved_at: str
    tier: int
    confidence: float
    source_type: str = "unknown"
    reporting_period: str | None = None
    as_of: str | None = None

    def comparable(self) -> str:
        """Normalised form used to decide whether two values agree.

        Delegates to :func:`src.text.fold` so that identity matching and
        conflict detection can never disagree about whether two spellings
        denote the same value.
        """
        return fold(self.value)


@dataclass
class Conflict:
    """A field with two or more materially different asserted values."""

    field: str
    values: list[FieldValue]
    resolution: str
    resolved_value: Any = None
    reporting_period: str | None = None
    note: str = ""

    def summary(self) -> str:
        rendered = " | ".join(
            f"{v.value} ({v.source_type}, tier {v.tier}, {v.source_url})"
            for v in self.values
        )
        return f"{self.field}: {rendered} -> {self.resolution}"


@dataclass
class Resolution:
    """The outcome of reconciling every claim about one field."""

    field: str
    value: Any
    winner: FieldValue | None
    claims: list[FieldValue] = field(default_factory=list)
    conflict: Conflict | None = None

    @property
    def has_conflict(self) -> bool:
        return self.conflict is not None


def _tier_outranks(a: FieldValue, b: FieldValue) -> bool:
    """True when ``a`` should outrank ``b`` on source authority alone."""
    if a.tier != b.tier:
        return a.tier < b.tier
    return a.confidence > b.confidence


def reconcile(
    field_name: str,
    claims: Iterable[FieldValue],
) -> Resolution:
    """Reconcile every claim about one field into a single resolution.

    Resolution order, highest authority first:

    1. An official registry record (Tier 1) is dispositive.
    2. Otherwise the most authoritative non-registry claim wins.
    3. Genuine ties are reported as an unresolved conflict rather than
       decided arbitrarily.
    """
    claims = [c for c in claims if c is not None]
    if not claims:
        return Resolution(field=field_name, value=None, winner=None)

    # Most authoritative first; stable within a tier.
    ordered = sorted(claims, key=lambda c: (c.tier, -c.confidence))

    best = ordered[0]
    normalised = {c.comparable() for c in claims if c.value is not None}

    if len(normalised) <= 1:
        return Resolution(field=field_name, value=best.value, winner=best, claims=ordered)

    # Values genuinely differ. A Tier-1 registry claim settles it.
    registry = [c for c in claims if c.tier == 1]
    if registry:
        winner = registry[0]
        return Resolution(
            field=field_name,
            value=winner.value,
            winner=winner,
            claims=ordered,
            conflict=Conflict(
                field=field_name,
                values=ordered,
                resolution="resolved_by_official_registry",
                resolved_value=winner.value,
                note=(
                    f"{len(ordered)} sources disagree; Brreg is "
                    f"dispositive and reports {winner.value!r}"
                ),
            ),
        )

    # No registry record to arbitrate. Prefer the highest-tier claim, but
    # only if it strictly outranks every dissenting claim.
    dissenting = [c for c in ordered if c.comparable() != best.comparable()]
    if dissenting and _tier_outranks(best, dissenting[0]):
        return Resolution(
            field=field_name,
            value=best.value,
            winner=best,
            claims=ordered,
            conflict=Conflict(
                field=field_name,
                values=ordered,
                resolution="resolved_by_source_tier",
                resolved_value=best.value,
                note=(
                    f"sources disagree; tier {best.tier} "
                    f"({best.source_type}) outranks tier "
                    f"{dissenting[0].tier} ({dissenting[0].source_type})"
                ),
            ),
        )

    return Resolution(
        field=field_name,
        value=best.value,
        winner=best,
        claims=ordered,
        conflict=Conflict(
            field=field_name,
            values=ordered,
            resolution="unresolved_sources_disagree",
            note=(
                "sources of comparable authority disagree; publishing the "
                "highest-confidence value and flagging the disagreement"
            ),
        ),
    )


def reconcile_all(
    claims_by_field: dict[str, list[FieldValue]],
) -> dict[str, Resolution]:
    return {
        name: reconcile(name, claims)
        for name, claims in claims_by_field.items()
    }


def report_field(resolution: Resolution) -> str:
    """Split a composite reconciliation key back into field and period.

    Claims are grouped under ``"net_result@2025"`` so that figures for
    different fiscal years are never mistaken for disagreement about the
    same figure.
    """
    name = resolution.field
    if "@" in name:
        return name.rsplit("@", 1)[0]
    return name


def conflict_report(
    resolutions: dict[str, Resolution],
) -> list[dict[str, Any]]:
    """A compact, publishable description of every disagreement."""
    report = []
    for name, resolution in resolutions.items():
        if not resolution.has_conflict:
            continue
        conflict = resolution.conflict
        assert conflict is not None
        report.append(
            {
                "field": report_field(resolution),
                "reporting_period": _period_of(name),
                "resolution": conflict.resolution,
                "note": conflict.note,
                "published_value": resolution.value,
                "values": [
                    {
                        "value": v.value,
                        "source_url": v.source_url,
                        "source_type": v.source_type,
                        "tier": v.tier,
                        "confidence": v.confidence,
                        "reporting_period": v.reporting_period,
                    }
                    for v in conflict.values
                ],
            }
        )
    return report


def _period_of(key: str) -> str | None:
    return key.rsplit("@", 1)[1] if "@" in key else None
