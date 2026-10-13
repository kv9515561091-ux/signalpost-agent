"""
Brønnøysundregistrene (Brreg) — Enhetsregisteret client.

This is the authoritative identity source for the agent. Everything the
agent later discovers about a company must be checked against the record
returned here, so this module deliberately surfaces the *whole* entity
record rather than a convenient subset.

Endpoint (no authentication required, NLOD licence):
    https://data.brreg.no/enhetsregisteret/api/enheter/{orgnr}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.cache import cached_json
from src.http import NotFound
from src.orgnumber import validate

ENHETSREGISTERET_BASE = "https://data.brreg.no/enhetsregisteret/api"

REQUEST_TIMEOUT = 20


class CompanyNotFound(NotFound):
    """The org number is well-formed but has no record in Enhetsregisteret."""


# How the most recent identity fetch was served. Lets the caller record
# that a profile came from cache, or was served stale, instead of
# presenting either as a fresh read.
LAST_FETCH_META: dict[str, Any] = {}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_address(address: dict[str, Any] | None) -> str | None:
    """Render a Brreg address block as a single human-readable line.

    Norwegian addresses pair the postal code with the post town without a
    separator — ``0180 OSLO`` — so the code and town are joined with a
    space while the street and locality are separated by commas.

    >>> format_address({"adresse": ["Akersgata 64"], "postnummer": "0180",
    ...                 "poststed": "OSLO", "land": "Norge"})
    'Akersgata 64, 0180 OSLO, Norge'
    """
    if not address:
        return None

    street = ", ".join(address.get("adresse") or [])

    postal_code = address.get("postnummer")
    post_town = address.get("poststed")
    locality = " ".join(p for p in (postal_code, post_town) if p)

    rendered = ", ".join(
        part for part in (street, locality, address.get("land")) if part
    )
    return rendered or None


def _code_block(block: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalise a Brreg ``{kode, beskrivelse}`` classification block."""
    if not block:
        return None
    return {
        "code": block.get("kode"),
        "description": block.get("beskrivelse"),
    }


def _historical_names(names: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalise the entity's former legal names.

    Useful for catching sources that still refer to the company by an old
    name — a common false-negative source during identity verification.
    """
    if not names:
        return []
    return [
        {
            "name": n.get("navn"),
            "from": n.get("fraDato"),
            "to": n.get("tilDato"),
        }
        for n in names
    ]


def _entity_url(org_number: str) -> str:
    return f"{ENHETSREGISTERET_BASE}/enheter/{org_number}"


def fetch_raw(org_number: str) -> dict[str, Any]:
    """Fetch the untouched Brreg record. Used by tests and fixtures.

    Identity is the one call that must not degrade: without it there is no
    company to describe, so failures propagate after a retry rather than
    yielding an empty record.
    """
    org_number = validate(org_number)
    body, meta = cached_json(_entity_url(org_number))
    LAST_FETCH_META.update(meta)
    return body


def get_company(company_number: str) -> dict[str, Any]:
    """Resolve a Norwegian organization number to its canonical record.

    Returns a normalised dictionary containing every field the agent can
    publish with Tier-1 confidence, plus ``source`` and ``retrieved_at``
    for evidence attribution.

    Raises:
        InvalidCompanyNumber: Malformed number or failed MOD-11 checksum.
        CompanyNotFound: No such entity.
    """
    org_number = validate(company_number)

    try:
        data = fetch_raw(org_number)
    except NotFound as exc:
        raise CompanyNotFound(
            f"No Enhetsregisteret record for organization number {org_number}"
        ) from exc

    postadresse = format_address(data.get("postadresse"))
    forretningsadresse = format_address(data.get("forretningsadresse"))

    return {
        "company_number": data.get("organisasjonsnummer"),
        "company_name": data.get("navn"),
        "previous_names": _historical_names(data.get("historiskeNavn")),

        # Legal form and status
        "organization_form_code": (data.get("organisasjonsform") or {}).get("kode"),
        "organization_form": (data.get("organisasjonsform") or {}).get(
            "beskrivelse"
        ),
        "status": "sletted" if data.get("sletted") else "aktiv",
        "bankruptcy": data.get("konkurs"),
        "bankruptcy_date": data.get("konkursdato"),
        "under_liquidation": data.get("underAvvikling"),
        "under_liquidation_date": data.get("underAvviklingDato"),
        "under_forced_liquidation": data.get(
            "underTvangsavviklingEllerTvangsopplosning"
        ),

        # Dates
        "registration_date": data.get("registreringsdatoEnhetsregisteret"),
        "founding_date": data.get("stiftelsesdato"),
        "articles_date": data.get("vedtektsdato"),
        "last_filed_annual_accounts": data.get("sisteInnsendteAarsregnskap"),

        # Addresses
        "address": forretningsadresse or postadresse,
        "business_address": forretningsadresse,
        "postal_address": postadresse,
        "municipality": (
            data.get("forretningsadresse") or data.get("postadresse") or {}
        ).get("kommune"),
        "municipality_code": (
            data.get("forretningsadresse") or data.get("postadresse") or {}
        ).get("kommunenummer"),

        # Classification
        "industry": _code_block(data.get("naeringskode1")),
        "industry_secondary": _code_block(data.get("naeringskode2")),
        "sector": _code_block(data.get("institusjonellSektorkode")),

        # Scale
        "employee_count": data.get("antallAnsatte"),
        "employee_count_registered": data.get("harRegistrertAntallAnsatte"),
        "employee_count_as_of": data.get(
            "registreringsdatoAntallAnsatteEnhetsregisteret"
        ),

        # What the company says it does — first-party, authoritative
        "business_description": " ".join(data.get("aktivitet") or []) or None,
        "notations": data.get("paategninger") or [],

        # Contact
        "phone": data.get("telefon"),
        "mobile": data.get("mobil"),
        "email": data.get("epostadresse"),
        "website": data.get("hjemmeside"),
        "language_form": data.get("maalform"),

        # Group structure
        "is_group_parent": data.get("erIKonsern"),
        "parent_company_number": data.get("overordnetEnhet"),

        # Register memberships
        "registered_business_register": data.get(
            "registrertIForetaksregisteret"
        ),
        "registered_vat_register": data.get("registrertIMvaregisteret"),
        "registered_voluntary_register": data.get(
            "registrertIFrivillighetsregisteret"
        ),
        "registered_party_register": data.get("registrertIPartiregisteret"),
        "registered_foundation_register": data.get(
            "registrertIStiftelsesregisteret"
        ),

        "source": _entity_url(org_number),
        # When the body actually came from the origin, not from the cache.
        "retrieved_at": LAST_FETCH_META.get("retrieved_at") or _iso_now(),
        "from_cache": bool(LAST_FETCH_META.get("cached")),
        "stale": bool(LAST_FETCH_META.get("stale")),
    }
