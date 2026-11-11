"""Adapter for the published Signalpost output contract.

The research pipeline stores a rich profile.  The evaluator expects one
terminal envelope per input company, with a small set of stable fields and
source identifiers that can be checked independently.  This module keeps
that translation in one place so the runner cannot drift from the format.

The envelope is deliberately a superset.  ``OUTPUT_CONTRACT.md`` names
``run``, ``claims``, ``evidence`` and ``operations``; the starter's
``terminal_envelope`` names ``run_id``, ``state``, ``modules`` and
``profile``.  Both are emitted, and both validation vocabularies are kept
terminal.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_STATES = {
    "complete",
    "not_applicable",
    "not_found",
    "blocked_policy",
    "blocked_robots",
    "source_error",
    "budget_exhausted",
    "submission_error",
}

SOURCE_CLASSES = {
    "official_registry": "government_registry",
    "official_registry_roles": "government_registry",
    "official_subunits": "government_registry",
    "official_annual_accounts": "official_filing",
    "official_company_website": "company_owned",
    "official_job_feed": "third_party_database",
    "news_source": "news",
    "reputable_business_database": "third_party_database",
    "wikidata": "third_party_database",
    "official_lei_registry": "third_party_database",
    "search_result": "search_result",
    "unverified_web_content": "other",
}

# Claim availability vocabulary from OUTPUT_CONTRACT.md.  ``available``
# always comes from a sourced claim; the other states describe a source
# that was checked and produced nothing, or could not be checked.
CLAIM_STATES = {
    "not_available",
    "blocked",
    "not_applicable",
    "ambiguous",
    "failed",
}


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with the contract's ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digits(value: Any) -> str:
    return "".join(character for character in str(value or "") if character.isdigit())


def _value_from_record(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in (
        "organisation_number",
        "organization_number",
        "company_number",
        "org_number",
        "orgnr",
    ):
        if value.get(key) is not None:
            return value[key]
    return None


def _json_values(path: Path) -> list[Any]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        for key in (
            "organisation_numbers",
            "organization_numbers",
            "company_numbers",
            "companies",
            "organisations",
        ):
            values = body.get(key)
            if isinstance(values, list):
                return values
    raise ValueError(
        f"{path} must contain a JSON list or an object with an "
        "organisation-number list"
    )


def read_inputs(path: str | Path) -> list[dict[str, Any]]:
    """Read one organisation number per input record.

    Accepts the starter's JSON, JSONL and plain-text forms.  Optional
    ``evaluation_split`` and ``sample_slice`` fields are preserved so a
    local report can keep the evaluator's split labels.

    Raises:
        ValueError: if an entry is not a nine-digit organisation number or
            the file contains duplicates.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    suffix = source.suffix.lower()
    if suffix == ".json":
        values = _json_values(source)
    elif suffix == ".jsonl":
        values = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        values = [
            line.strip()
            for line in source.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]

    records: list[dict[str, Any]] = []
    for value in values:
        raw_value = _value_from_record(value)
        org = _digits(raw_value)
        if len(org) != 9:
            raise ValueError(f"Invalid Norwegian organisation number: {value!r}")
        record: dict[str, Any] = {"organisation_number": org}
        if isinstance(value, dict):
            for key in ("evaluation_split", "sample_slice"):
                if value.get(key) is not None:
                    record[key] = value[key]
        records.append(record)

    organisations = [record["organisation_number"] for record in records]
    duplicates = sorted(
        organisation
        for organisation in set(organisations)
        if organisations.count(organisation) > 1
    )
    if duplicates:
        raise ValueError(
            "Organisation-number input contains duplicates: "
            + ", ".join(duplicates)
        )
    return records


def source_class(source_type: str | None) -> str:
    """Map the agent's source taxonomy to the contract's source classes."""
    return SOURCE_CLASSES.get(str(source_type or ""), "other")


def _span(item: dict[str, Any]) -> str:
    text = item.get("evidence_text")
    if text:
        return str(text)
    value = item.get("value")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value if value is not None else "")


def _content_sha256(item: dict[str, Any], span: str) -> str:
    """Hash the recorded source span.

    The profile does not retain full page bytes, so this is the digest of
    the exact excerpt the claim was taken from, not of an unrecorded page.
    A caller that captured the full snapshot can pass ``snapshot_sha256``
    and that digest is preferred.
    """
    snapshot = str(item.get("snapshot_sha256") or "")
    if snapshot:
        return snapshot
    material = span
    if not material:
        material = f"{item.get('source_url', '')}\n{item.get('retrieved_at', '')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


_EXTRACTION_METHODS = {
    "official_registry": "api-json",
    "official_registry_roles": "api-json",
    "official_subunits": "csv-snapshot",
    "official_annual_accounts": "pdf-text",
    "official_company_website": "html-text",
    "official_job_feed": "api-json",
    "news_source": "html-text",
    "reputable_business_database": "html-text",
    "wikidata": "api-json",
    "official_lei_registry": "api-json",
    "search_result": "search-result",
    "unverified_web_content": "html-text",
}


def _extraction_method(item: dict[str, Any]) -> str:
    explicit = item.get("extraction_method")
    if explicit:
        return str(explicit)
    source_type = str(item.get("source_type") or "")
    return _EXTRACTION_METHODS.get(source_type, "html-text")


def _snapshot(
    item: dict[str, Any],
    *,
    source_url: str,
    content_sha256: str,
    span: str,
) -> dict[str, Any]:
    """The immutable-source reference a reviewer can re-check.

    The raw bytes live in the crawl cache keyed by URL; this record pins
    what was read: URL, retrieval time, content hash and extraction
    method.  Two runs over the same snapshot produce the same id.
    """
    return {
        "id": f"snap-{content_sha256[:16]}",
        "source_url": source_url,
        "retrieved_at": item.get("retrieved_at"),
        "content_sha256": content_sha256,
        "extraction_method": _extraction_method(item),
        "claim_span": span,
    }


def _evidence_id(organisation_number: str, item: dict[str, Any], span: str) -> str:
    """A stable evidence id derived from what the evidence says.

    Positional ids (``ev-org-0001``) shift whenever extraction order
    changes, which makes a rerun look like new evidence.  Hashing the
    claim's identity keeps ids stable across refreshes.
    """
    material = "\x1f".join(
        (
            str(item.get("field") or ""),
            str(item.get("source_url") or ""),
            span,
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"ev-{organisation_number}-{digest}"


def _availability(value: Any) -> str:
    if value is None or value == "" or value == [] or value == {}:
        return "not_available"
    return "available"


def _evidence_for_profile(
    organisation_number: str,
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return contract claims and evidence for one profile."""
    raw_items: list[dict[str, Any]] = []
    for key in ("evidence", "facts"):
        values = profile.get(key)
        if isinstance(values, list):
            raw_items.extend(item for item in values if isinstance(item, dict))

    claims: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    for item in raw_items:
        span = _span(item)
        base_id = _evidence_id(organisation_number, item, span)
        occurrence = seen_ids.get(base_id, 0) + 1
        seen_ids[base_id] = occurrence
        evidence_id = base_id if occurrence == 1 else f"{base_id}-{occurrence}"
        source_url = str(item.get("source_url") or "")
        source_type = str(item.get("source_type") or "")
        content_sha256 = _content_sha256(item, span)
        record = {
            "id": evidence_id,
            "source_url": source_url,
            "source_class": source_class(source_type),
            "retrieved_at": item.get("retrieved_at"),
            "content_sha256": content_sha256,
            "claim_span": span,
            "field": item.get("field"),
            "source_type": source_type,
            "source_tier": item.get("source_tier"),
            "confidence": item.get("confidence"),
            "extraction_method": _extraction_method(item),
            "snapshot": _snapshot(
                item,
                source_url=source_url,
                content_sha256=content_sha256,
                span=span,
            ),
        }
        evidence.append(record)

        claim = {
            "field": item.get("field"),
            "value": item.get("value"),
            "availability": _availability(item.get("value")),
            "confidence": item.get("confidence"),
            "evidence_ids": [evidence_id],
            "source_url": source_url,
            "retrieved_at": item.get("retrieved_at"),
        }
        if item.get("as_of") is not None:
            claim["as_of"] = item["as_of"]
        if item.get("reporting_period") is not None:
            claim["reporting_period"] = item["reporting_period"]
        claims.append(claim)

    # Families that were checked and produced nothing must say so.  A
    # missing row reads as "never looked"; an explicit state is checkable.
    claimed_fields = {claim["field"] for claim in claims}
    availability = profile.get("availability")
    if isinstance(availability, dict):
        for field_name, state in sorted(availability.items()):
            if field_name in claimed_fields or state not in CLAIM_STATES:
                continue
            claims.append(
                {
                    "field": field_name,
                    "value": None,
                    "availability": state,
                    "confidence": 0.0,
                    "evidence_ids": [],
                    "source_url": None,
                    "retrieved_at": None,
                }
            )

    return claims, evidence


def _errors(profile: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if profile.get("error"):
        errors.append(
            {
                "code": "profile_error",
                "message": str(profile["error"]),
            }
        )
    for message in profile.get("retrieval_errors") or []:
        errors.append(
            {
                "code": "retrieval_error",
                "message": str(message),
            }
        )
    return errors


def _module_state(
    profile: dict[str, Any],
    *,
    name: str,
    present: bool,
    errors: list[str],
) -> dict[str, Any]:
    prefix = f"{name}:"
    matching = [error for error in errors if error.startswith(prefix)]
    if profile.get("error"):
        state = "submission_error"
    elif matching:
        state = "source_error"
    elif present:
        state = "complete"
    else:
        state = "not_found"
    return {
        "state": state,
        "retry_count": 0,
        "final_timestamp": profile.get("retrieved_at"),
    }


def _modules(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    errors = [str(error) for error in profile.get("retrieval_errors") or []]
    website_checked = bool(
        profile.get("website")
        or profile.get("website_verified") is not None
        or profile.get("website_verification")
    )
    modules: dict[str, dict[str, Any]] = {
        "registry": _module_state(
            profile,
            name="registry",
            present=bool(profile.get("company_name")),
            errors=errors,
        ),
        "roles": _module_state(
            profile,
            name="roles",
            present=bool(profile.get("roles") or profile.get("ceo_name")),
            errors=errors,
        ),
        "financials": _module_state(
            profile,
            name="financials",
            present=bool(profile.get("financials")),
            errors=errors,
        ),
        "website": {
            "state": (
                "submission_error"
                if profile.get("error")
                else "complete"
                if profile.get("website")
                else "not_found"
                if website_checked
                else "not_applicable"
            ),
            "retry_count": 0,
            "final_timestamp": profile.get("retrieved_at"),
        },
    }
    if profile.get("locations"):
        modules["locations"] = _module_state(
            profile,
            name="locations",
            present=True,
            errors=errors,
        )
    return modules


def _organisation_number(profile: dict[str, Any]) -> str:
    for key in ("company_number", "organisation_number", "organization_number"):
        org = _digits(profile.get(key))
        if org:
            return org
    return ""


# The information families a reviewer checks, and the claim fields that
# satisfy them. Prefix entries match any field that starts with them.
_COVERAGE_FAMILIES: dict[str, tuple[str, ...]] = {
    "identity": (
        "company_name",
        "legal_form",
        "industry",
        "industry_code",
        "address",
        "municipality",
        "registration_date",
        "founding_date",
    ),
    "contact": ("website", "phone", "email"),
    "social": ("social_",),
    "leadership": ("ceo_name", "chair_name", "roles", "role:"),
    "financials": (
        "revenue",
        "operating_result",
        "net_result",
        "total_assets",
        "total_equity",
        "operating_margin",
        "equity_ratio",
    ),
    "workforce": (
        "employee_count",
        "workplace_employee_count",
        "jobs_count",
        "job_opening",
    ),
    "locations": ("locations", "workplace_"),
    "activity": ("recent_activity",),
}

# What a reader should look at first. Ordered by how much it says about
# the company's current state.
_HIGHLIGHT_FIELDS = (
    "jobs_count",
    "job_opening",
    "recent_activity",
    "revenue",
    "net_result",
    "employee_count",
    "ceo_name",
)


def _family_matches(field: str, patterns: tuple[str, ...]) -> bool:
    return any(
        field == pattern or (pattern.endswith(":") and field.startswith(pattern))
        or (pattern.endswith("_") and field.startswith(pattern))
        for pattern in patterns
    )


def _synthesis(
    profile: dict[str, Any],
    *,
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """The reviewer-facing read of the record, above the raw claims.

    Beyond restating the profile this states what was *not* found, which
    sources carried the record, and which information families the run
    actually covered. A reader can judge the record without opening the
    claims, and an empty family is visible instead of implied.
    """
    explanation = profile.get("explanation")
    text = ""
    generated_by = "deterministic"
    if isinstance(explanation, dict):
        text = str(explanation.get("text") or "")
        generated_by = str(explanation.get("generated_by") or "deterministic")

    changes = profile.get("changes")
    changed = len(changes) if isinstance(changes, list) else 0
    if changed:
        changes_summary = f"{changed} field(s) changed since the previous run"
    else:
        changes_summary = "No changes recorded since the previous run"

    available: list[dict[str, Any]] = [
        claim for claim in claims if claim.get("availability") == "available"
    ]
    states = Counter(str(claim.get("availability")) for claim in claims)

    families: list[dict[str, Any]] = []
    present_families: list[str] = []
    missing_families: list[str] = []
    for family, patterns in _COVERAGE_FAMILIES.items():
        hits = [
            claim
            for claim in available
            if _family_matches(str(claim.get("field") or ""), patterns)
        ]
        if hits:
            present_families.append(family)
        else:
            missing_families.append(family)
        families.append(
            {
                "family": family,
                "claims": len(hits),
                "fields": sorted(
                    {str(claim.get("field")) for claim in hits}
                ),
            }
        )

    unknowns = [
        {"field": str(claim.get("field")), "state": claim.get("availability")}
        for claim in claims
        if claim.get("availability") != "available"
    ]

    by_class = Counter(str(item.get("source_class")) for item in evidence)
    types_by_class: dict[str, set[str]] = {}
    for item in evidence:
        types_by_class.setdefault(str(item.get("source_class")), set()).add(
            str(item.get("source_type"))
        )
    sources = [
        {
            "source_class": name,
            "claims": count,
            "source_types": sorted(types_by_class.get(name, set())),
        }
        for name, count in sorted(by_class.items())
    ]

    highlights: list[str] = []
    for field_name in _HIGHLIGHT_FIELDS:
        for claim in available:
            if claim.get("field") != field_name:
                continue
            value = claim.get("value")
            if value in (None, "", [], {}):
                continue
            highlights.append(f"{field_name}: {value}")
            break
        if len(highlights) >= 3:
            break

    return {
        "summary": text,
        "highlights": highlights,
        "changes_summary": changes_summary,
        "generated_by": generated_by,
        "claim_count": len(claims),
        "claim_states": dict(sorted(states.items())),
        "sources": sources,
        "coverage": {
            "available_families": present_families,
            "missing_families": missing_families,
            "families": families,
        },
        "unknowns": unknowns,
    }


def envelope_for(
    profile: dict[str, Any],
    *,
    run_id: str,
    started_at: str,
    completed_at: str,
) -> dict[str, Any]:
    """Convert one persisted profile or error record into a terminal envelope."""
    organisation_number = _organisation_number(profile)
    claims, evidence = _evidence_for_profile(organisation_number, profile)
    errors = _errors(profile)
    failed = bool(profile.get("error"))
    operations = profile.get("_operations")
    if not isinstance(operations, dict):
        operations = {}

    return {
        "organisation_number": organisation_number,
        "run": {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "terminal_status": "failed" if failed else "completed",
        },
        "claims": claims,
        "evidence": evidence,
        "synthesis": _synthesis(profile, claims=claims, evidence=evidence),
        "changes": profile.get("changes") or [],
        "errors": errors,
        "operations": {
            "requests": int(operations.get("requests") or 0),
            "runtime_ms": int(operations.get("runtime_ms") or 0),
            "third_party_cost_usd": float(
                operations.get("third_party_cost_usd") or 0.0
            ),
        },
        # Starter-compatible fields.  Keeping the whole profile here also
        # preserves rejected extraction candidates and retrieval errors
        # for a reviewer without polluting the contract arrays.
        "run_id": run_id,
        "state": "submission_error" if failed else "complete",
        "started_at": started_at,
        "completed_at": completed_at,
        "modules": _modules(profile),
        "profile": profile,
        "evaluation_split": profile.get("evaluation_split"),
        "sample_slice": profile.get("sample_slice"),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise


def write_jsonl(envelopes: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write one compact JSON object per line, atomically."""
    lines = [
        json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        for envelope in envelopes
    ]
    _atomic_write(Path(path), "\n".join(lines) + ("\n" if lines else ""))


def write_json(value: Any, path: str | Path) -> None:
    """Write one indented JSON document atomically."""
    _atomic_write(
        Path(path),
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
    )


def _terminal(state: Any) -> bool:
    return state in TERMINAL_STATES


def build_report(
    envelopes: list[dict[str, Any]],
    *,
    expected_count: int,
    run_id: str,
    started_at: str,
    completed_at: str,
    runtime_ms: int,
    requests: int,
    input_path: str,
) -> dict[str, Any]:
    """Validate the returned envelopes against the documented invariants."""
    organisations = [envelope.get("organisation_number") for envelope in envelopes]
    unique_organisations = len(organisations) == len(set(organisations))
    module_states = [
        state.get("state")
        for envelope in envelopes
        for state in (envelope.get("modules") or {}).values()
        if isinstance(state, dict)
    ]
    checks = {
        "exact_expected_count": len(envelopes) == expected_count,
        "unique_organisation_numbers": unique_organisations,
        "all_entity_states_terminal": all(
            _terminal(envelope.get("state")) for envelope in envelopes
        ),
        "all_module_states_terminal": all(
            _terminal(state) for state in module_states
        ),
        "zero_silent_drops": (
            len(envelopes) == expected_count and unique_organisations
        ),
        "every_envelope_has_organisation_number": all(organisations),
    }
    states: dict[str, int] = {}
    for envelope in envelopes:
        state = str(envelope.get("state") or "unknown")
        states[state] = states.get(state, 0) + 1

    return {
        "run_id": run_id,
        "input": input_path,
        "started_at": started_at,
        "completed_at": completed_at,
        "runtime_ms": runtime_ms,
        "requests": requests,
        "expected_count": expected_count,
        "returned_count": len(envelopes),
        "unique_count": len(set(organisations)),
        "committed_claims": sum(
            len(envelope.get("claims") or []) for envelope in envelopes
        ),
        "committed_evidence": sum(
            len(envelope.get("evidence") or []) for envelope in envelopes
        ),
        "states": states,
        "checks": checks,
        "passed": all(checks.values()),
    }


def company_rows(envelopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A compact per-company audit index for the report."""
    return [
        {
            "organisation_number": envelope.get("organisation_number"),
            "state": envelope.get("state"),
            "claims": len(envelope.get("claims") or []),
            "evidence": len(envelope.get("evidence") or []),
            "errors": len(envelope.get("errors") or []),
        }
        for envelope in envelopes
    ]


def missing_results(
    inputs: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Error records for inputs the pipeline failed to return."""
    returned = {
        _organisation_number(result)
        for result in results
        if _organisation_number(result)
    }
    return [
        {
            "company_number": record["organisation_number"],
            "error": "SubmissionError: no profile returned for input",
        }
        for record in inputs
        if record["organisation_number"] not in returned
    ]


__all__ = [
    "SOURCE_CLASSES",
    "TERMINAL_STATES",
    "build_report",
    "company_rows",
    "envelope_for",
    "missing_results",
    "read_inputs",
    "source_class",
    "utc_now",
    "write_json",
    "write_jsonl",
]
