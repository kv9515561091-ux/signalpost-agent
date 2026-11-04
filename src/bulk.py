"""Frozen Brreg bulk snapshot reader.

The official run hands over a batch of company numbers but not the full
Enhetsregisteret CSV.  When a snapshot is available, reading identity from
it removes three live registry requests per company and is the difference
between a 1,000-company run finishing and hitting the 2,000-request ceiling
around company 667.

Only requested rows are retained.  The export has roughly 411,000 rows, so
streaming it once is cheap in memory and still gives the report a full-file
SHA-256 for provenance.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import hashlib
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

import requests

from src.request_tracker import request_tracker

BULK_URL = "https://data.brreg.no/enhetsregisteret/api/enheter/lastned/csv"
DEFAULT_BULK_PATH = Path("data/brreg_bulk.csv.gz")
UNDERENHETER_URL = (
    "https://data.brreg.no/enhetsregisteret/api/underenheter/lastned/csv"
)
DEFAULT_UNDERENHETER_PATH = Path("data/brreg_underenheter.csv.gz")


def _first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _bool(value: str, *, default: bool | None = False) -> bool | None:
    text = (value or "").strip().casefold()
    if not text:
        return default
    return text in {"1", "true", "ja", "yes"}


def _int(value: str) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _address(row: dict[str, str], prefix: str) -> str | None:
    street = _first(row, f"{prefix}.adresse")
    postal = _first(row, f"{prefix}.postnummer")
    town = _first(row, f"{prefix}.poststed")
    country = _first(row, f"{prefix}.land")
    locality = " ".join(part for part in (postal, town) if part)
    rendered = ", ".join(
        part for part in (street, locality, country) if part
    )
    return rendered or None


def _classification(row: dict[str, str], prefix: str) -> dict[str, str | None] | None:
    code = _first(row, f"{prefix}.kode")
    description = _first(row, f"{prefix}.beskrivelse")
    if not code and not description:
        return None
    return {"code": code or None, "description": description or None}


def normalize_row(
    row: dict[str, str],
    *,
    retrieved_at: str,
    source: str = BULK_URL,
) -> dict[str, Any]:
    """Map a Brreg CSV row to the shape ``build_profile`` reads."""
    company_number = _first(row, "organisasjonsnummer", "Organisasjonsnummer")
    industry = _classification(row, "naeringskode1")
    sector = _classification(row, "institusjonellSektorkode")
    form_code = _first(row, "organisasjonsform.kode")
    form_name = _first(row, "organisasjonsform.beskrivelse")
    business_address = _address(row, "forretningsadresse")
    postal_address = _address(row, "postadresse")

    return {
        "company_number": company_number,
        "company_name": _first(row, "navn"),
        "previous_names": [],
        "organization_form_code": form_code or None,
        "organization_form": form_name or form_code or None,
        "status": "sletted" if _bool(_first(row, "sletted")) else "aktiv",
        "bankruptcy": _bool(_first(row, "konkurs")),
        "under_liquidation": _bool(_first(row, "underAvvikling")),
        "registration_date": _first(
            row,
            "registreringsdatoEnhetsregisteret",
            "registreringsdatoenhetsregisteret",
        ) or None,
        "founding_date": _first(row, "stiftelsesdato") or None,
        "last_filed_annual_accounts": _first(
            row, "sisteInnsendteAarsregnskap"
        ) or None,
        "address": business_address or postal_address,
        "business_address": business_address,
        "postal_address": postal_address,
        "municipality": _first(
            row, "forretningsadresse.kommune", "postadresse.kommune"
        ) or None,
        "municipality_code": _first(
            row,
            "forretningsadresse.kommunenummer",
            "postadresse.kommunenummer",
        ) or None,
        "industry": industry,
        "sector": sector,
        "employee_count": _int(_first(row, "antallAnsatte")),
        "employee_count_as_of": _first(
            row, "registreringsdatoAntallAnsatteEnhetsregisteret"
        ) or None,
        "business_description": _first(row, "aktivitet", "vedtektsfestetFormaal")
        or None,
        "phone": _first(row, "telefon") or None,
        "email": _first(row, "epostadresse") or None,
        "website": _first(row, "hjemmeside") or None,
        "is_group_parent": _bool(_first(row, "erIKonsern"), default=None),
        "parent_company_number": _first(row, "overordnetEnhet") or None,
        "source": source,
        "retrieved_at": retrieved_at,
        "from_cache": False,
        "stale": False,
    }


def normalize_subunit(
    row: dict[str, str],
    *,
    retrieved_at: str,
    source: str = UNDERENHETER_URL,
) -> dict[str, Any]:
    """Map a Brreg underenhet row to a workplace record."""
    workplace_address = _address(row, "beliggenhetsadresse")
    postal_address = _address(row, "postadresse")
    return {
        "organisation_number": _first(row, "organisasjonsnummer") or None,
        "name": _first(row, "navn") or None,
        "organisation_form": _first(row, "organisasjonsform.beskrivelse")
        or _first(row, "organisasjonsform.kode")
        or None,
        "address": workplace_address or postal_address,
        "municipality": _first(
            row,
            "beliggenhetsadresse.kommune",
            "postadresse.kommune",
        )
        or None,
        "municipality_code": _first(
            row,
            "beliggenhetsadresse.kommunenummer",
            "postadresse.kommunenummer",
        )
        or None,
        "industry": _classification(row, "naeringskode1"),
        "employee_count": _int(_first(row, "antallAnsatte")),
        "employee_count_as_of": _first(
            row, "registreringsdatoAntallAnsatteEnhetsregisteret"
        )
        or None,
        "website": _first(row, "hjemmeside") or None,
        "email": _first(row, "epostadresse") or None,
        "phone": _first(row, "telefon") or None,
        "registration_date": _first(
            row, "registreringsdatoIEnhetsregisteret"
        )
        or None,
        "parent_company_number": _first(row, "overordnetEnhet") or None,
        "source": source,
        "retrieved_at": retrieved_at,
    }


@contextlib.contextmanager
def _open_text(path: Path) -> Iterator[TextIO]:
    with path.open("rb") as probe:
        gzipped = probe.read(2) == b"\x1f\x8b"
    if gzipped:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield handle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_snapshot(url: str, path: str | Path) -> Path:
    destination = Path(path)
    if destination.is_file() and destination.stat().st_size > 1_000_000:
        with destination.open("rb") as handle:
            if handle.read(2) == b"\x1f\x8b":
                return destination
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    request_tracker.record()
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30, 900),
            headers={
                "Accept": "application/gzip, application/octet-stream, */*",
                "Accept-Encoding": "identity",
            },
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        with temporary.open("rb") as handle:
            if handle.read(2) != b"\x1f\x8b":
                raise ValueError("bulk export was not gzip-compressed")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def ensure_bulk(path: str | Path = DEFAULT_BULK_PATH) -> Path:
    """Return a local identity snapshot, downloading it once when absent.

    A 1,000-company run cannot afford three live registry calls per
    company: the 2,000-request ceiling is crossed around company 667.
    The public export is one request and removes the identity call.
    """
    return _ensure_snapshot(BULK_URL, path)


def ensure_underenheter(path: str | Path = DEFAULT_UNDERENHETER_PATH) -> Path:
    """Return a local underenhet snapshot, downloading it once when absent.

    Subunits carry registered workplaces.  Reading them from the export
    adds workplaces without spending a request per company, which the
    live ``/underenheter`` endpoint would.
    """
    return _ensure_snapshot(UNDERENHETER_URL, path)


def load_bulk(
    path: str | Path,
    organisation_numbers: Iterable[str],
    *,
    retrieved_at: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load requested identity records from a Brreg CSV snapshot.

    Returns the found records keyed by organisation number plus metadata
    for the run report.  The caller decides whether missing rows are a
    hard error; a partial local snapshot is otherwise hard to diagnose.
    """
    source = Path(path)
    wanted = {str(value) for value in organisation_numbers}
    found: dict[str, dict[str, Any]] = {}
    scanned = 0

    snapshot_sha256 = _sha256(source)

    with _open_text(source) as handle:
        header = handle.readline()
        handle.seek(0)
        delimiter = max(";,\t", key=header.count)
        for row in csv.DictReader(handle, delimiter=delimiter):
            scanned += 1
            org = _first(row, "organisasjonsnummer", "Organisasjonsnummer")
            if len(org) != 9 or org not in wanted:
                continue
            found[org] = normalize_row(
                row,
                retrieved_at=retrieved_at,
                source=BULK_URL,
            )
            if len(found) == len(wanted):
                break

    return found, {
        "path": str(source),
        "snapshot_sha256": snapshot_sha256,
        "rows_scanned": scanned,
        "requested": len(wanted),
        "selected": len(found),
        "missing": sorted(wanted - set(found)),
    }


def load_underenheter(
    path: str | Path,
    parent_numbers: Iterable[str],
    *,
    retrieved_at: str,
    limit_per_parent: int = 25,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load registered workplaces keyed by their parent organisation.

    The export has roughly half a million rows.  Only rows whose
    ``overordnetEnhet`` is in the requested set are retained, and each
    parent is capped so one large group cannot dominate the envelope.
    """
    source = Path(path)
    wanted = {str(value) for value in parent_numbers}
    found: dict[str, list[dict[str, Any]]] = {}
    scanned = 0
    workplaces = 0
    cap = max(1, limit_per_parent)

    snapshot_sha256 = _sha256(source)

    with _open_text(source) as handle:
        header = handle.readline()
        handle.seek(0)
        delimiter = max(";,\t", key=header.count)
        for row in csv.DictReader(handle, delimiter=delimiter):
            scanned += 1
            parent = _first(row, "overordnetEnhet")
            if len(parent) != 9 or parent not in wanted:
                continue
            bucket = found.setdefault(parent, [])
            if len(bucket) >= cap:
                continue
            bucket.append(
                normalize_subunit(row, retrieved_at=retrieved_at)
            )
            workplaces += 1
            if len(found) == len(wanted) and all(
                len(items) >= cap for items in found.values()
            ):
                break

    return found, {
        "path": str(source),
        "snapshot_sha256": snapshot_sha256,
        "rows_scanned": scanned,
        "requested": len(wanted),
        "selected": len(found),
        "workplaces": workplaces,
    }


__all__ = [
    "BULK_URL",
    "DEFAULT_BULK_PATH",
    "DEFAULT_UNDERENHETER_PATH",
    "UNDERENHETER_URL",
    "ensure_bulk",
    "ensure_underenheter",
    "load_bulk",
    "load_underenheter",
    "normalize_row",
    "normalize_subunit",
]
