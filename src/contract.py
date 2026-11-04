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
    "news_source": "news",
    "reputable_business_database": "third_party_database",
    "search_result": "search_result",
    "unverified_web_content": "other",
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
    """
    material = span
    if not material:
        material = f"{item.get('source_url', '')}\n{item.get('retrieved_at', '')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


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
    for index, item in enumerate(raw_items, start=1):
        evidence_id = f"ev-{organisation_number}-{index:04d}"
        span = _span(item)
        source_url = str(item.get("source_url") or "")
        source_type = str(item.get("source_type") or "")
        record = {
            "id": evidence_id,
            "source_url": source_url,
            "source_class": source_class(source_type),
            "retrieved_at": item.get("retrieved_at"),
            "content_sha256": _content_sha256(item, span),
            "claim_span": span,
            "field": item.get("field"),
            "source_type": source_type,
            "source_tier": item.get("source_tier"),
            "confidence": item.get("confidence"),
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
