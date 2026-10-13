"""
Brønnøysundregistrene — Rolleksregisteret (management and board).

Roles are served from the Enhetsregisteret namespace and require no
authentication:

    https://data.brreg.no/enhetsregisteret/api/enheter/{orgnr}/roller

The response groups roles by *role group* (``DAGL`` managing director,
``STYR`` board, ``REVI`` auditor) and then lists the individual roles
inside each group. A person appears under many role codes — the group
tells you what kind of body they sit on, the role code tells you what
they do on it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.brreg import ENHETSREGISTERET_BASE
from src.cache import cached_json
from src.orgnumber import validate

REQUEST_TIMEOUT = 20

# Role groups worth surfacing as distinct profile facts.
ROLE_GROUPS_OF_INTEREST = {
    "DAGL": "managing_director",
    "STYR": "board",
    "REVI": "auditor",
    "INNH": "owner",
    "DAGLSTYR": "managing_director_and_board",
}


def _roles_url(org_number: str) -> str:
    # The path segment is "roller" (Norwegian), not "roles". Getting this
    # wrong returns 404 for every company, which the fault tolerance below
    # turns into a silent "this company has no management" — so the exact
    # URL is pinned by a test.
    return f"{ENHETSREGISTERET_BASE}/enheter/{org_number}/roller"


def _empty(url: str, retrieved_at: str, error: str) -> dict[str, Any]:
    """A well-formed empty result that records why it is empty."""
    return {
        "roles": [],
        "groups": [],
        "managing_directors": [],
        "board": [],
        "auditors": [],
        "owners": [],
        "ceo_name": None,
        "board_chair_name": None,
        "board_size": 0,
        "auditor_name": None,
        "owner_name": None,
        "source": url,
        "retrieved_at": retrieved_at,
        "error": error,
    }


def _person_name(person: dict[str, Any]) -> str:
    name = person.get("navn") or {}
    parts = [
        name.get("fornavn"),
        name.get("mellomnavn"),
        name.get("etternavn"),
    ]
    return " ".join(p for p in parts if p).strip()


def _parse_role(role: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise one role entry into a flat record."""
    if role.get("avregistrert"):
        return None

    role_type = (role.get("type") or {}).get("kode")
    role_label = (role.get("type") or {}).get("beskrivelse")

    person = role.get("person")
    entity = role.get("enhet")

    record: dict[str, Any] = {
        "role_code": role_type,
        "role": role_label,
    }

    if person:
        record["person_name"] = _person_name(person)
        # Birth year is public in the register and is a useful
        # disambiguator when two people share a name.
        birthdate = person.get("fodselsdato")
        if birthdate:
            record["birth_year"] = str(birthdate)[:4]
        if person.get("erDoed"):
            record["deceased"] = True

    elif entity:
        record["entity_number"] = entity.get("organisasjonsnummer")
        record["entity_name"] = " ".join(entity.get("navn") or [])

    else:
        return None

    return record


def get_roles(company_number: str) -> dict[str, Any]:
    """Fetch management, board and auditor records for an organization.

    Returns a dict with a flat ``roles`` list plus convenience keys for the
    roles most commonly needed in a company profile. Entities with no roles
    on file return empty collections rather than raising.

    Management is an enrichment, not the identity anchor. A transient
    network failure therefore yields an empty result carrying an ``error``
    key rather than discarding the company profile that the caller has
    already assembled from the registry.
    """
    org_number = validate(company_number)
    url = _roles_url(org_number)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    # fetch_json retries once; a persistent failure degrades to an empty
    # result that records why, rather than discarding the profile.
    try:
        body, meta = cached_json(url, accept_missing=True)
        data = body or {}
    except Exception as exc:  # noqa: BLE001
        return _empty(url, retrieved_at, f"{type(exc).__name__}: {exc}")

    data = data or {}

    roles: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    for group in data.get("rollegrupper") or []:
        group_type = (group.get("type") or {}).get("kode")

        group_records = [
            parsed
            for parsed in (_parse_role(r) for r in group.get("roller") or [])
            if parsed
        ]

        if not group_records:
            continue

        for record in group_records:
            record["group_code"] = group_type
            record["group"] = ROLE_GROUPS_OF_INTEREST.get(
                group_type, (group.get("type") or {}).get("beskrivelse")
            )
            record["group_changed_at"] = group.get("sistEndret")

        roles.extend(group_records)
        groups.append(
            {
                "group_code": group_type,
                "group": (group.get("type") or {}).get("beskrivelse"),
                "changed_at": group.get("sistEndret"),
                "count": len(group_records),
            }
        )

    managing_directors = [
        r for r in roles if r.get("group_code") == "DAGL"
    ]
    board = [r for r in roles if r.get("group_code") == "STYR"]
    auditors = [r for r in roles if r.get("group_code") == "REVI"]
    owners = [r for r in roles if r.get("group_code") == "INNH"]

    return {
        "roles": roles,
        "groups": groups,

        "managing_directors": managing_directors,
        "board": board,
        "auditors": auditors,
        "owners": owners,

        # Single most useful summary values.
        "ceo_name": (
            managing_directors[0].get("person_name")
            if managing_directors
            else None
        ),
        "board_chair_name": next(
            (
                r.get("person_name")
                for r in board
                if r.get("role_code") == "LEDE"
            ),
            None,
        ),
        "board_size": len(board),
        "auditor_name": next(
            (
                r.get("entity_name") or r.get("person_name")
                for r in auditors
            ),
            None,
        ),

        # An ENK has no managing director: the sole proprietor is the
        # company. Surfacing the owner prevents "no CEO" from reading as
        # "no accountable person".
        "owner_name": next(
            (r.get("person_name") for r in owners if r.get("person_name")),
            None,
        ),

        "source": url,
        "retrieved_at": retrieved_at,
    }
