"""
Wikidata exact-entity resolution for a whole batch in one request.

Wikidata is the cheapest external identity source available: a single
SPARQL query resolves every organisation number in the batch, so the
connector costs one request no matter how many companies are checked.

The registry is already the authority for name, industry and address, so
nothing here republishes those. What Wikidata adds is an external
identifier that proves the exact legal entity was resolved -- the QID and,
where present, the LEI -- plus a website and employee count only when the
registry left them empty. Coverage is low (a few percent of the corpus),
which is why it is a single batch call rather than a per-company lookup.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"

# WDQS answers a bare client with a 403. The policy asks for a descriptive
# agent string with a contact.
USER_AGENT = (
    "SignalpostAgent/1.0 (+company-intelligence; contact: agent@example.no)"
)

# One VALUES clause of 1,000 nine-digit strings is about 13 KB of query
# text, which WDQS handles in roughly ten seconds.
BATCH_SIZE = 1000

# Norwegian organisation number (P2333). The registry number is the
# entity key, so only that property is required; everything else is an
# OPTIONAL attribute.
_QUERY = """SELECT ?org ?item
  (GROUP_CONCAT(DISTINCT ?leiValue; separator="|") AS ?lei)
  (GROUP_CONCAT(DISTINCT ?websiteValue; separator="|") AS ?website)
  (GROUP_CONCAT(DISTINCT STR(?employeesValue); separator="|") AS ?employees)
WHERE {
  VALUES ?org { %s }
  ?item wdt:P2333 ?org .
  OPTIONAL { ?item wdt:P1278 ?leiValue . }
  OPTIONAL { ?item wdt:P856 ?websiteValue . }
  OPTIONAL { ?item wdt:P1128 ?employeesValue . }
}
GROUP BY ?org ?item
"""


@dataclass(frozen=True)
class WikidataEntity:
    """What one organisation number resolved to on Wikidata."""

    org_number: str
    qid: str
    lei: str | None = None
    website: str | None = None
    employees: int | None = None

    @property
    def url(self) -> str:
        return f"https://www.wikidata.org/wiki/{self.qid}"


def normalise_org_numbers(values: Iterable[str]) -> list[str]:
    """Keep only 9-digit organisation numbers, de-duplicated, in order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        digits = "".join(character for character in str(value) if character.isdigit())
        if len(digits) != 9 or digits in seen:
            continue
        seen.add(digits)
        ordered.append(digits)
    return ordered


def sparql_query(
    org_numbers: Sequence[str],
    *,
    endpoint: str = DEFAULT_ENDPOINT,
) -> str:
    """The POST body that asks WDQS for every entity in ``org_numbers``.

    ``endpoint`` is accepted so the caller can point at a mirror, and so
    tests can assert the request shape without a network.
    """
    values = " ".join(f'"{number}"' for number in org_numbers)
    return _QUERY % values


def _first(value: str | None) -> str | None:
    if not value:
        return None
    for part in value.split("|"):
        part = part.strip()
        if part:
            return part
    return None


def _as_int(value: str | None) -> int | None:
    first = _first(value)
    if first is None:
        return None
    try:
        parsed = int(float(first))
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _qid_from_uri(uri: str) -> str | None:
    text = (uri or "").rstrip("/")
    if "/" not in text:
        return None
    candidate = text.rsplit("/", 1)[-1]
    return candidate or None


def parse_results(payload: dict) -> dict[str, WikidataEntity]:
    """Turn a SPARQL JSON result into entities keyed by organisation number.

    A single org number can resolve to more than one item; the row that
    carries an LEI wins, because an identifier is stronger evidence than a
    bare QID. Otherwise the first row is kept.
    """
    rows = (
        payload.get("results", {}).get("bindings", [])
        if isinstance(payload, dict)
        else []
    )

    found: dict[str, WikidataEntity] = {}
    for row in rows:
        org = (row.get("org") or {}).get("value")
        qid = _qid_from_uri((row.get("item") or {}).get("value") or "")
        if not org or not qid:
            continue
        entity = WikidataEntity(
            org_number=str(org),
            qid=qid,
            lei=_first((row.get("lei") or {}).get("value")),
            website=_first((row.get("website") or {}).get("value")),
            employees=_as_int((row.get("employees") or {}).get("value")),
        )
        current = found.get(entity.org_number)
        if current is None or (entity.lei and not current.lei):
            found[entity.org_number] = entity

    return found


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def resolve(
    org_numbers: Iterable[str],
    *,
    fetch: Callable[[str], dict],
    batch_size: int = BATCH_SIZE,
) -> dict[str, WikidataEntity]:
    """Resolve every organisation number, one request per batch.

    ``fetch`` takes a SPARQL query string and returns the decoded JSON
    payload, so the caller owns request counting, POST semantics and the
    User-Agent.
    """
    numbers = normalise_org_numbers(org_numbers)
    resolved: dict[str, WikidataEntity] = {}
    for chunk in _chunks(numbers, batch_size):
        payload = fetch(sparql_query(chunk))
        resolved.update(parse_results(payload))
    return resolved


def identity_facts(
    entity: WikidataEntity,
    *,
    retrieved_at: str,
) -> list[dict]:
    """Fact records for the external identity Wikidata supplies.

    The QID and LEI are identifiers, so the evidence span is the
    identifier itself as it appears on the Wikidata page.
    """
    facts: list[dict] = [
        {
            "field": "wikidata_id",
            "value": entity.qid,
            "source_url": entity.url,
            "retrieved_at": retrieved_at,
            "evidence_text": entity.qid,
            "confidence": 0.7,
            "source_tier": 4,
            "source_type": "wikidata",
        }
    ]

    if entity.lei:
        facts.append(
            {
                "field": "lei",
                "value": entity.lei,
                "source_url": entity.url,
                "retrieved_at": retrieved_at,
                "evidence_text": entity.lei,
                "confidence": 0.75,
                "source_tier": 4,
                "source_type": "wikidata",
            }
        )

    return facts
