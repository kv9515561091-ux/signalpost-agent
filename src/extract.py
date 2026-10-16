"""
Schema-guided extraction from verified pages.

The registries supply identity, management and financials. They do not
supply what a company actually sells, who its customers are, or what it
announced last month — and those are a large share of the information a
company profile should carry.

An LLM reads the gap, but it does not get to decide what is true. Every
fact it proposes must arrive with a ``quote``: a span copied verbatim
from the page. The quote is then located in the page text the agent
actually fetched, and a fact whose quote cannot be found is discarded.

That turns hallucination from something invisible into something
detectable. It does not eliminate it — a quote can be real and still be
lifted from a passage about a different company, which is why pages
must already have passed identity verification before reaching here, and
why an organization number belonging to another entity is rejected
outright.

Fields the registry owns are excluded from the schema entirely. The
model is never asked to restate identity, so it can never overwrite it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from src import llm
from src.sources import TIER_MAX_CONFIDENCE
from src.text import fold

# Facts the registry already supplies, authoritatively. The model is
# never asked for these, so it cannot contradict or overwrite them.
RESERVED_FIELDS = frozenset({
    "company_number",
    "company_name",
    "legal_form",
    "status",
    "registration_date",
    "founding_date",
    "address",
    "business_address",
    "postal_address",
    "municipality",
    # Address restated as free text. A live run spent a model call
    # rediscovering the registered address, which Brreg had already
    # supplied for free; the slot is better spent on something the
    # registry cannot tell us.
    "locations",
    "geographic_presence",
    "industry",
    "industry_code",
    "employee_count",
    "website",
    "phone",
    "email",
    "bankruptcy",
    "under_liquidation",
    "ceo_name",
    "owner_name",
    "board_chair_name",
    "auditor_name",
    "board_size",
    "revenue",
    "net_result",
    "total_assets",
    "total_equity",
    "operating_result",
    "operating_costs",
    "profit_before_tax",
})

# The vocabulary the model is allowed to fill, chosen to match the
# taxonomy in the specification while staying clear of what Brreg
# already holds. Every entry here is something the registry cannot tell
# us, which is the only reason to spend a model call on it.
EXTRACTABLE_FIELDS = (
    "business_summary",
    "products",
    "services",
    "target_customers",
    "markets_served",
    "certifications",
    "technology_stack",
    "ownership_structure",
    "recent_developments",
    "partnerships",
    "sustainability_initiatives",
)

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": list(EXTRACTABLE_FIELDS),
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "The fact, stated concisely in English."
                        ),
                    },
                    "quote": {
                        "type": "string",
                        "description": (
                            "A span copied VERBATIM from the page "
                            "containing the fact. This is checked against "
                            "the page and the fact is discarded if it "
                            "cannot be located. Do not paraphrase."
                        ),
                    },
                    "published_date": {
                        "type": "string",
                        "description": (
                            "Date the page or news item was published, "
                            "as YYYY-MM-DD, or empty if not stated."
                        ),
                    },
                },
                "required": ["field", "value", "quote", "published_date"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You extract structured facts about a company from one web page.

The page has already been verified as belonging to the target company by
matching its organization number. Treat its content as authoritative
about that company.

Rules:
- Copy the `quote` field VERBATIM from the page. Do not paraphrase,
  translate, reformat or reconstruct it. A quote that cannot be found in
  the page causes the fact to be discarded.
- Extract only what the page actually states. If the page does not say
  something, do not report it. An empty `facts` array is a correct and
  acceptable answer.
- Do not report facts about any other organization, person or product
  mentioned on the page, even if they appear in a list or comparison.
- Do not report figures that the company itself registers with the
  Norwegian business registry (revenue, profit, employee count, legal
  form, address, municipality, locations). Those are already sourced
  authoritatively from the registry, and repeating them wastes the only
  thing you were called for.
- Report only what this page says that a registry lookup could not.
  A description of the business, the things it sells or does, the
  customers it serves, the certifications it holds, and dated
  announcements are all worth reporting. An address or a legal form is
  not.
- Report facts about the BUSINESS. Biographical details about the
  individuals behind it — family, children, health, hobbies, personal
  history, where someone used to live — are not company facts, even
  when the page is the company's own site and the detail is accurate.
  A page that mixes a CV with a company description is a source for the
  second half only.
- If a page is a personal site that mentions a business only in passing,
  report the business and nothing else.
- Prefer specific, verifiable statements over vague marketing language.
- If the page describes a service, product or event, date it using the
  date on the page if one is given.

Choosing the `field`:
- `business_summary` is the right field for a great many facts. When in
  doubt between a broad field and a narrow one, prefer `business_summary`.
  A statement filed under the wrong narrow field is worse than one filed
  under the broad one, because a reader sees `products` and expects a
  product.
- Do not classify a headline, a news item or a link label as `products`,
  `services`, `partnerships` or `recent_developments` unless the text
  actually states what the product is, what the service does, who the
  partners are, or what happened. A title like "Equinor and Shell
  partner on the Bay du Nord project" is a `recent_developments` item at
  best; "Analyst call" is not a fact about the business at all.
- If the page genuinely only supports `business_summary`, filing
  everything there is correct. Do not stretch a sentence into a narrow
  field to make the output look varied.
"""

_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# Norwegian pages state dates as 12.03.2025 and 12. mars 2025 far more
# often than as ISO, so a bare ISO match would discard most real dates.
_NORWEGIAN_MONTHS = {
    "januar": 1, "februar": 2, "mars": 3, "april": 4,
    "mai": 5, "juni": 6, "juli": 7, "august": 8,
    "september": 9, "oktober": 10, "november": 11, "desember": 12,
}
_DATE_DOTTED = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_DATE_LONG = re.compile(
    r"\b(\d{1,2})\.?\s+([A-Za-zæøåÆØÅ]+)\s+(\d{4})\b", re.IGNORECASE
)
_ORG_NUMBER = re.compile(r"(?<!\d)\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)")


def _tokens(value: str) -> set[str]:
    """Distinctive tokens for comparing two fact statements.

    Numbers are kept at any length. "148 countries" and "52 countries"
    differ only in that number, and dropping short numeric tokens made
    genuinely distinct facts look identical.
    """
    return {
        token
        for token in fold(value).split()
        if len(token) >= 4 or token.isdigit()
    }


def _overlap(a: str, b: str) -> float:
    """How much of the shorter statement is covered by the longer.

    Containment rather than Jaccard: two phrasings of one fact often
    differ in length, and Jaccard penalises exactly the rephrasing we
    want to catch. The score is 1.0 when every distinctive token of the
    shorter fact appears in the longer one.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# Personal-biography signals, Norwegian and English.
#
# The business-only rule is an instruction to the model, and instructions
# are the weakest link in this pipeline: the quote check proves a sentence
# exists on the page, not that the sentence is about the company. A page
# that mixes a founder's CV with a company description passes identity,
# passes the quote check, and then contributes the CV.
#
# These patterns are deliberately narrow. Each requires a term that has
# essentially no place in a business description, so that "family-owned"
# or "healthcare" -- both plausible in a real company profile -- are not
# caught. A false positive here discards a legitimate fact, so anything
# ambiguous is left to the prompt.
_PERSONAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:hustru|ektefelle|gift)\b", re.I), "spouse"),
    (re.compile(r"\b(?:barn|barnas|barnebarn|sønn|datter)\b", re.I), "children"),
    (re.compile(r"\b(?:født|født på|født i)\b", re.I), "birth"),
    (re.compile(r"\b(?:døde|dødsfall|begravet|minnesmerke)\b", re.I), "death"),
    (re.compile(
        r"\b(?:hobby|hobbyer|fritid|seilmer|sykkelmer|jakthund)\b", re.I,
    ), "hobbies"),
    (re.compile(
        r"\b(?:gift med|forelsket|skilt|skiltet|bryllup)\b", re.I,
    ), "marital status"),
    (re.compile(
        r"\b(?:spouse|wife|husband|married to|divorced)\b", re.I,
    ), "spouse"),
    (re.compile(r"\b(?:his|her)\s+(?:wife|husband|son|daughter|mother)\b", re.I),
     "family"),
    (re.compile(r"\b(?:born in|birthday|grew up in)\b", re.I), "biography"),
    (re.compile(r"\b(?:was born|passed away|in memoriam)\b", re.I), "biography"),
    (re.compile(r"\b(?:obituary|funeral service)\b", re.I), "death"),
    (re.compile(
        r"\b(?:survived by|preceded in death|his death|her death)\b", re.I,
    ), "death"),
)

# The full quote must be almost entirely personal for it to be refused.
# A company profile that mentions one founder's spouse in passing is
# still a company profile.
# A *distinct* personal signal is sufficient; density is not the test.
#
# Density was tried first and refused real biography. "He is married to
# Anna and they have three children" scores 2 personal words across 10 --
# only 0.2 -- because English pronouns and prepositions are doing the
# work, and the actual signal ("children") is one word. Tuning the
# threshold down to catch it dragged "our founder's wife Anna led the
# charity" (1 word, 0.10) into territory where a company mentioning a
# spouse once would be discarded.
#
# Density is still used, but only to distinguish an incidental mention,
# which yields exactly one signal and reads as a clause inside a longer
# business statement, from text that is *about* a person.
_PERSONAL_RATIO_THRESHOLD = 0.18


def _is_personal_content(value: str, quote: str) -> str | None:
    """Whether a fact is biography rather than company information.

    Returns the signal name when refused, or ``None`` when the fact looks
    like business content.

    Scored against the quote when one is available, because that is the
    text the fact is claiming to be drawn from, falling back to the value.
    """
    text = quote or value
    folded = fold(text)

    if not folded:
        return None

    hits = 0
    for pattern, _label in _PERSONAL_PATTERNS:
        if pattern.search(text):
            hits += 1

    if not hits:
        return None

    # Require the personal terms to account for a real share of the text
    # before refusing, so a single incidental mention survives.
    personal_words = sum(
        len(m.group(0).split())
        for pattern, _ in _PERSONAL_PATTERNS
        for m in pattern.finditer(text)
    )
    total_words = max(1, len(text.split()))

    if personal_words / total_words < _PERSONAL_RATIO_THRESHOLD:
        return None

    return "personal biography rather than company information"


def _restates_registry(
    value: str, registry_values: Iterable[str | None]
) -> bool:
    """Whether a fact merely repeats something Brreg already supplied.

    Directory sites reprint registry fields. Publishing those again as
    if they were new discoveries wastes a slot and inflates coverage with
    noise. The field allowlist cannot stop this on its own, because a
    model asked for ``business_summary`` will happily tuck an employee
    count in there.
    """
    candidate = fold(value)
    if not candidate:
        return False

    for known in registry_values:
        if not known:
            continue
        normalised = fold(known)
        if len(normalised) < 6:
            continue
        if normalised and normalised in candidate:
            return True

    return False


@dataclass
class ExtractedFact:
    """One proposed fact, after verification."""

    field: str
    value: str
    quote: str
    source_url: str
    retrieved_at: str
    tier: int
    source_type: str
    confidence: float
    published_date: str | None = None
    verification: str = "quote_matched"
    notes: list[str] = dataclass_field(default_factory=list)


@dataclass
class ExtractionReport:
    """What extraction attempted and what survived verification."""

    source_url: str
    facts: list[ExtractedFact] = dataclass_field(default_factory=list)
    rejected: list[dict[str, Any]] = dataclass_field(default_factory=list)
    llm_used: bool = False
    error: str | None = None
    # Whether a second, breadth-seeking attempt was made, and what the
    # first one refused. Recorded so the extra model call is visible
    # rather than invisible cost.
    retried_for_breadth: bool = False
    retried_rejections: list[dict[str, Any]] = dataclass_field(
        default_factory=list
    )

    @property
    def verified_count(self) -> int:
        return len(self.facts)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def quote_exists(quote: str, page_text: str) -> bool:
    """Whether ``quote`` appears in ``page_text``.

    Compares on whitespace-normalised, diacritic-folded text so that a
    model reformatting line breaks or retyping Norwegian letters does not
    cause a real quote to be rejected, while genuinely absent or
    paraphrased text still fails.
    """
    if not quote or not quote.strip():
        return False

    if _normalise_whitespace(quote) in _normalise_whitespace(page_text):
        return True

    return fold(quote) in fold(page_text)


def _valid_date(day: int, month: int, year: int) -> str | None:
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2200):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _valid_published_date(value: str | None) -> str | None:
    """Normalise a stated publication date to ISO, or ``None``.

    Accepts ISO (``2025-03-12``), numeric Norwegian (``12.03.2025``) and
    long-form Norwegian (``12. mars 2025``). Anything unrecognised is
    dropped rather than guessed at: a wrong date is worse than no date,
    because it would feed the freshness assessment.
    """
    if not value:
        return None

    text = str(value)

    match = _DATE.search(text)
    if match:
        return _valid_date(int(match.group(3)), int(match.group(2)),
                           int(match.group(1)))

    match = _DATE_DOTTED.search(text)
    if match:
        return _valid_date(int(match.group(1)), int(match.group(2)),
                           int(match.group(3)))

    match = _DATE_LONG.search(text)
    if match:
        month = _NORWEGIAN_MONTHS.get(fold(match.group(2)).replace(" ", ""))
        if month:
            return _valid_date(int(match.group(1)), month, int(match.group(3)))

    return None


def verify_candidate(
    candidate: dict[str, Any],
    *,
    page_text: str,
    source_url: str,
    retrieved_at: str,
    tier: int,
    source_type: str,
    company_number: str,
    registry_values: Iterable[str | None] = (),
    accepted: Iterable[ExtractedFact] = (),
) -> tuple[ExtractedFact | None, str | None]:
    """Check one model-proposed fact.

    Returns ``(fact, None)`` when it is publishable, or ``(None, reason)``
    describing precisely why it was refused.
    """
    name = candidate.get("field")
    value = candidate.get("value")
    quote = candidate.get("quote")

    # Registry-owned fields are checked first so the refusal explains the
    # real reason: the model is not offered these at all, so proposing
    # one means it invented the idea or copied it from elsewhere.
    if name in RESERVED_FIELDS:
        return None, f"field is owned by the registry: {name}"
    if name not in EXTRACTABLE_FIELDS:
        return None, f"field not extractable: {name}"
    if not value or not str(value).strip():
        return None, "empty value"

    # quote_exists() rejects a missing or blank quote, so once it returns
    # True the value is a real str and the later .strip() is safe.
    if not quote_exists(quote or "", page_text):
        return None, "quote not found in page"
    quote = str(quote)

    # A quote that cites a different organization is describing another
    # entity, whatever else it says.
    for cited in _ORG_NUMBER.findall(quote or ""):
        digits = "".join(ch for ch in cited if ch.isdigit())
        if digits != company_number:
            return None, f"quote cites another organization: {digits}"

    text = str(value).strip()

    if _restates_registry(text, registry_values):
        return None, "repeats a value already sourced from the registry"

    # Biography that happens to sit on a verified page is still not a
    # company fact. Checked before deduplication so the recorded reason
    # is the substantive one.
    personal = _is_personal_content(text, quote)
    if personal:
        return None, personal

    for existing in accepted:
        if existing.field == name and _overlap(existing.value, text) >= 0.5:
            return None, f"duplicates an accepted {name} fact"

    return (
        ExtractedFact(
            field=name,
            value=text,
            quote=quote.strip(),
            source_url=source_url,
            retrieved_at=retrieved_at,
            tier=tier,
            source_type=source_type,
            confidence=TIER_MAX_CONFIDENCE.get(tier, 0.25),
            published_date=_valid_published_date(
                candidate.get("published_date")
            ),
        ),
        None,
    )


def extract_from_page(
    *,
    page_text: str,
    source_url: str,
    retrieved_at: str,
    company_number: str,
    company_name: str,
    tier: int,
    source_type: str,
    title: str = "",
    registry_values: Iterable[str | None] = (),
    already_accepted: Iterable[ExtractedFact] = (),
) -> ExtractionReport:
    """Extract and verify facts from one page.

    Args:
        already_accepted: Facts already published from earlier pages.
            Two directories frequently state the same thing, and without
            carrying these forward each page would pass its own copy
            through independently.

    Never raises: a model failure yields a report whose ``error`` says so
    and whose ``facts`` list is empty.
    """
    report = ExtractionReport(source_url=source_url)
    # Carried forward so cross-page duplicates are caught here rather
    # than surfacing twice in the profile.
    seen_facts = list(already_accepted)

    if not page_text or not page_text.strip():
        report.error = "page had no text content"
        return report

    if not llm.is_available():
        report.error = llm.unavailable_reason() or "LLM unavailable"
        return report

    header = f"Company: {company_name} (org.nr {company_number})"
    if title:
        header += f"\nPage title: {title}"
    header += f"\nURL: {source_url}\n\n--- PAGE CONTENT ---\n{page_text}"

    result = _call_model(header)
    if result is None:
        report.error = _last_error or "extraction failed"
        return report

    report.llm_used = True
    _verify_result(
        result,
        report,
        seen_facts,
        page_text=page_text,
        source_url=source_url,
        retrieved_at=retrieved_at,
        tier=tier,
        source_type=source_type,
        company_number=company_number,
        registry_values=registry_values,
    )

    # One retry, and only for the failure that looked like an empty page
    # but might not have been: every proposal refused as a registry
    # restatement. Costs a model call, not a request.
    if not report.facts and _restatements_only(report.rejected):
        second = _call_model(header + _BREADTH_HINT)
        if second is not None:
            # A second attempt at an already-declined page has to be
            # allowed to succeed at the same claims, so the refusals from
            # the first pass are dropped rather than carried forward --
            # otherwise every useful fact is refused a second time as a
            # duplicate of something nothing published.
            first_pass_rejections = report.rejected
            report.rejected = []
            report.facts = []
            _verify_result(
                second,
                report,
                seen_facts,
                page_text=page_text,
                source_url=source_url,
                retrieved_at=retrieved_at,
                tier=tier,
                source_type=source_type,
                company_number=company_number,
                registry_values=registry_values,
            )
            report.retried_for_breadth = True
            report.retried_rejections = first_pass_rejections

    return report


# Asked when a first pass produced nothing but registry restatements.
#
# The measured failure: sixteen of the twenty companies that yielded no
# facts at all had every proposal refused for "repeats a value already
# sourced from the registry". For a single-product company whose website
# says essentially what Brreg says, that refusal is usually correct -- but
# it is also what a page that *does* describe products, customers and
# service areas looks like when the model stops at the first field it
# recognises. Asking again, told explicitly what has already been
# rejected and why, costs one model call and no additional request.
_BREADTH_HINT = """

The previous attempt produced only statements that repeat data the
registry already holds (name, address, legal form, and similar). Those
are rejected, so do not report them again.

Look past the headline. This page may still describe what the company
sells, who it supplies, which areas or markets it serves, how it works,
what it is certified or approved for, how many people it has, or what it
has done recently. Any of those would be new information even though the
company name is not.

If genuinely nothing here goes beyond the registry record, return an
empty `facts` array. That is a correct answer and will not be penalised.
"""


def _restatements_only(rejected: list[dict[str, Any]]) -> bool:
    """Whether every refusal was a registry restatement."""
    if not rejected:
        return False
    return all(
        "registry" in str(item.get("reason", "")).lower() for item in rejected
    )


# Set per-call because extract_from_page is not re-entrant across
# threads, and threading a mutable through six helpers is worse than a
# module-level cell the tests reset.
_last_error: str | None = None


def _call_model(header: str):
    """One model call, returning its result or ``None`` on any refusal."""
    global _last_error

    try:
        return llm.complete_json(EXTRACTION_SCHEMA, system=SYSTEM_PROMPT, user=header)
    except llm.LLMUnavailable as exc:
        _last_error = str(exc)
    except llm.LLMRefused as exc:
        _last_error = f"model refused: {exc}"
    except llm.LLMTruncated as exc:
        _last_error = f"response truncated: {exc}"
    except llm.LLMCallFailed as exc:
        _last_error = str(exc)
    return None


def _verify_result(
    result,
    report: ExtractionReport,
    seen_facts: list,
    *,
    page_text: str,
    source_url: str,
    retrieved_at: str,
    tier: int,
    source_type: str,
    company_number: str,
    registry_values: Iterable[str | None],
) -> ExtractionReport:
    """Check each proposed fact and record the refusals."""
    for candidate in (result.data or {}).get("facts") or []:
        if not isinstance(candidate, dict):
            report.rejected.append(
                {"candidate": str(candidate)[:200], "reason": "not an object"}
            )
            continue

        fact, reason = verify_candidate(
            candidate,
            page_text=page_text,
            source_url=source_url,
            retrieved_at=retrieved_at,
            tier=tier,
            source_type=source_type,
            company_number=company_number,
            registry_values=registry_values,
            accepted=seen_facts,
        )

        if fact is not None:
            report.facts.append(fact)
            seen_facts.append(fact)
        else:
            report.rejected.append(
                {
                    "field": candidate.get("field"),
                    "value": str(candidate.get("value"))[:200],
                    "quote": str(candidate.get("quote"))[:300],
                    "reason": reason,
                }
            )

    return report
