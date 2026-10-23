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

import copy
import os
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any
from urllib.parse import urlparse

from src import llm
from src.sources import TIER_FIRST_PARTY, TIER_MAX_CONFIDENCE
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
                            "Publication date visible in the quoted text, "
                            "as YYYY-MM-DD. Required for "
                            "recent_developments. Never use a modified or "
                            "updated timestamp."
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
- `recent_developments` only for a real announcement. Copy its visible
  publication date into the `quote` and set `published_date` to that
  date. Never use a "last updated", "modified", feed "updated" or footer
  copyright date. Do not report menu labels such as "News", "See more
  news" or "Les mer", and do not report product, category or listing
  pages as news.

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

# Two pages per call halves the number of calls, and the call, not the
# request, is the expensive part: a run is paced by a 30 RPM model
# limiter and each call waits on its own completion. Set to 1 to get the
# per-page path back for an A/B run.
MAX_BATCH_PAGES = max(1, int(os.environ.get("AGENT_BATCH_PAGES", "2")))

# A page can be 200,000 characters. Two of those in one prompt is both
# slow and close to the context limit, so anything larger than this is
# read on its own. The cap is generous: ordinary company pages are a
# few thousand characters, and the ones that exceed 50,000 are mostly
# navigation and scripts.
MAX_BATCH_PAGE_CHARS = 50_000

# Derived from SYSTEM_PROMPT rather than written twice, so a rule added
# there cannot silently fail to apply to batches. The changed sentences
# are the ones that assume a single page; each fact also has to say
# which page its quote came from.
BATCH_SYSTEM_PROMPT = (
    SYSTEM_PROMPT.replace(
        "You extract structured facts about a company from one web page.",
        "You extract structured facts about a company from several web "
        "pages at once.",
    )
    .replace(
        "The page has already been verified as belonging to the target "
        "company by\nmatching its organization number. Treat its content "
        "as authoritative\nabout that company.",
        "Each page has already been verified as belonging to the target "
        "company\nby matching its organization number. Treat each page's "
        "content as\nauthoritative about that company.",
    )
    .replace(
        "Copy the `quote` field VERBATIM from the page.",
        "Copy the `quote` field VERBATIM from the block named by `page`.",
    )
    + "\n- Every fact must carry the `page` id of the block its quote "
      "came from.\n  A quote that is not in the block it names is "
      "discarded.\n"
    "- Read every block before answering. Do not stop after the first "
    "block\n  because it is richer: a fact that appears only in the "
    "second block is still\n  wanted, and a page that was skipped "
    "contributes nothing.\n"
    "- Extract as thoroughly from each block as you would if it were "
    "the only\n  page. Do not compress, summarise, or trade detail in "
    "one block for\n  coverage of another. A content-rich block "
    "normally yields many facts;\n  finding only a handful means you "
    "stopped too early.\n"
)

_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# Norwegian pages state dates as 12.03.2025 and 12. mars 2025 far more
# often than as ISO, so a bare ISO match would discard most real dates.
_NORWEGIAN_MONTHS = {
    "januar": 1, "februar": 2, "mars": 3, "april": 4,
    "mai": 5, "juni": 6, "juli": 7, "august": 8,
    "september": 9, "oktober": 10, "november": 11, "desember": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "des": 12,
    "january": 1, "february": 2, "march": 3, "may": 5, "june": 6,
    "july": 7, "october": 10, "december": 12,
}
_MONTH_NAMES = {
    1: ("januar", "jan", "january"),
    2: ("februar", "feb", "february"),
    3: ("mars", "mar", "march"),
    4: ("april", "apr", "april"),
    5: ("mai", "mai", "may"),
    6: ("juni", "jun", "june"),
    7: ("juli", "jul", "july"),
    8: ("august", "aug", "august"),
    9: ("september", "sep", "september"),
    10: ("oktober", "okt", "october"),
    11: ("november", "nov", "november"),
    12: ("desember", "des", "december"),
}
_DATE_DOTTED = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_DATE_LONG = re.compile(
    r"\b(\d{1,2})\.?\s+([A-Za-z.]+)\s+(\d{4})\b", re.IGNORECASE
)
_DATE_MONTH_FIRST = re.compile(
    r"\b([A-Za-z.]+)\s+(\d{1,2}),?\s+(\d{4})\b", re.IGNORECASE
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


@dataclass
class PageInput:
    """One fetched page, as extraction sees it."""

    page_text: str
    source_url: str
    retrieved_at: str
    tier: int
    source_type: str
    title: str = ""


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

    Accepts ISO (``2025-03-12``), numeric Norwegian (``12.03.2025``),
    long-form Norwegian (``12. mars 2025``) and English month-first forms
    (``March 12, 2025``). Anything unrecognised is dropped rather than
    guessed at: a wrong date is worse than no date, because it would feed
    the freshness assessment.
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
        month = _NORWEGIAN_MONTHS.get(
            fold(match.group(2)).replace(" ", "").rstrip(".")
        )
        if month:
            return _valid_date(int(match.group(1)), month, int(match.group(3)))

    match = _DATE_MONTH_FIRST.search(text)
    if match:
        month = _NORWEGIAN_MONTHS.get(
            fold(match.group(1)).replace(" ", "").rstrip(".")
        )
        if month:
            return _valid_date(int(match.group(2)), month, int(match.group(3)))

    return None


_UPDATE_MARKERS = (
    "sist oppdatert", "sist endret", "sist endra",
    "last updated", "last modified", "modified on", "updated on",
    "oppdatert:", "updated:",
)
_UPDATE_WORD_NEAR_DATE = re.compile(
    r"\b(?:updated|modified|oppdatert)\b[\"']?\s*(?:on|:)?\s*[\"']?\s*"
    r"(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}\s+\d{1,2}\s+\d{4}"
    r"|\d{4}\s+\d{1,2}\s+\d{1,2}"
    r"|\d{1,2}\.?\s+[a-z]{3,9}\.?\s+\d{4}"
    r"|[a-z]{3,9}\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)


def _has_update_marker(text: str) -> bool:
    folded = fold(text)
    if any(marker in folded for marker in _UPDATE_MARKERS):
        return True
    return bool(_UPDATE_WORD_NEAR_DATE.search(folded))

_NEWS_NAVIGATION_LABELS = {
    "news", "nyheter", "nyhet", "siste nytt", "latest news",
    "more news", "all news", "all corporate news", "se mer", "se flere",
    "see more", "see more news", "read more", "les mer", "view all",
    "vis alle", "press", "presse", "media", "aktuelt", "events",
    "arrangementer", "products", "produkter", "tjenester", "services",
}

_NEWS_NAVIGATION_PREFIXES = (
    "se mer", "se flere", "see more", "read more", "les mer",
    "view all", "vis alle",
)

_PRODUCT_PATH_HINTS = (
    "/produkt", "/products", "/produkter", "/tjenester", "/services",
    "/butikk", "/shop", "/kategori", "/category", "/sortiment",
    "/kolleksjon", "/collections",
)


def _label(value: str) -> str:
    return " ".join(fold(value).replace("|", " ").split())


def _date_forms(iso: str) -> set[str]:
    year, month, day = (int(part) for part in iso.split("-"))
    forms = {
        iso,
        f"{day}.{month}.{year}",
        f"{day:02d}.{month:02d}.{year}",
    }
    for day_text in (str(day), f"{day:02d}"):
        for name in _MONTH_NAMES[month]:
            forms.update({
                f"{day_text}. {name} {year}",
                f"{day_text}. {name}. {year}",
                f"{day_text} {name} {year}",
                f"{day_text}.{name} {year}",
                f"{name} {day_text}, {year}",
            })
        if 10 <= day <= 19:
            ordinal = f"{day}th"
        elif day % 10 == 1:
            ordinal = f"{day}st"
        elif day % 10 == 2:
            ordinal = f"{day}nd"
        elif day % 10 == 3:
            ordinal = f"{day}rd"
        else:
            ordinal = f"{day}th"
        for name in _MONTH_NAMES[month]:
            forms.add(f"{name} {ordinal}, {year}")
    return forms


def _date_in_quote(published_date: str, quote: str) -> bool:
    folded_quote = fold(quote)
    return any(fold(form) in folded_quote for form in _date_forms(published_date))


def _looks_like_product_page(source_url: str) -> bool:
    path = urlparse(source_url).path.lower()
    return any(hint in path for hint in _PRODUCT_PATH_HINTS)


def _recent_development_reason(
    candidate: dict[str, Any],
    *,
    quote: str,
    source_url: str,
    published_date: str | None,
) -> str | None:
    if not published_date:
        return "recent developments require a stated publication date"

    if not _date_in_quote(published_date, quote):
        return "publication date is not present in the quoted text"

    if _has_update_marker(quote):
        return (
            "date is a modified or updated timestamp, not a publication "
            "date"
        )

    value_label = _label(str(candidate.get("value") or ""))
    quote_label = _label(quote)
    if (
        value_label in _NEWS_NAVIGATION_LABELS
        or quote_label in _NEWS_NAVIGATION_LABELS
        or value_label.startswith(_NEWS_NAVIGATION_PREFIXES)
    ):
        return "navigation or section label, not a news item"

    if _looks_like_product_page(source_url):
        return "product or listing page, not a dated news item"

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

    published_date = _valid_published_date(candidate.get("published_date"))
    if name == "recent_developments":
        reason = _recent_development_reason(
            candidate,
            quote=quote,
            source_url=source_url,
            published_date=published_date,
        )
        if reason:
            return None, reason

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
            published_date=published_date,
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
        report.error = getattr(_state, "error", None) or "extraction failed"
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
    #
    # The retry is earned when the page is the company speaking for
    # itself, or when the company has nothing yet and a directory may be
    # its only source. Once facts exist, a directory that only restated
    # the registry has nothing to add: measured across fifty companies,
    # directory pages yielded 0.11 facts each, and this second pass cost
    # as much as reading another page.
    if (
        not report.facts
        and _restatements_only(report.rejected)
        and (tier <= TIER_FIRST_PARTY or not seen_facts)
    ):
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


def extract_from_pages(
    pages: Iterable[PageInput],
    *,
    company_number: str,
    company_name: str,
    registry_values: Iterable[str | None] = (),
    already_accepted: Iterable[ExtractedFact] = (),
) -> list[ExtractionReport]:
    """Extract and verify facts from several pages in one model call.

    Returns one report per input page, in input order, so the caller can
    keep attributing facts, refusals and errors to the page they belong
    to. A page with no text is reported without a call; a batch that
    fails is retried as single pages, because a two-page prompt is the
    thing that failed, not the pages themselves.

    Never raises.
    """
    inputs = list(pages)
    reports = [ExtractionReport(source_url=page.source_url) for page in inputs]
    seen_facts = list(already_accepted)
    seen_before = list(already_accepted)

    usable: list[tuple[int, PageInput]] = []
    for index, page in enumerate(inputs):
        if not page.page_text or not page.page_text.strip():
            reports[index].error = "page had no text content"
            continue
        usable.append((index, page))

    if not usable:
        return reports

    if not llm.is_available():
        reason = llm.unavailable_reason() or "LLM unavailable"
        for index, _ in usable:
            reports[index].error = reason
        return reports

    if len(usable) == 1 or any(
        len(page.page_text) > MAX_BATCH_PAGE_CHARS for _, page in usable
    ):
        return _extract_each_individually(
            usable,
            reports,
            seen_facts,
            company_number=company_number,
            company_name=company_name,
            registry_values=registry_values,
        )

    ids = [f"p{number}" for number in range(1, len(usable) + 1)]
    pages_by_id = {pid: page for pid, (_, page) in zip(ids, usable, strict=True)}
    reports_by_id = {
        pid: reports[index] for pid, (index, _) in zip(ids, usable, strict=True)
    }

    result = _call_model(
        _batch_header(ids, pages_by_id, company_name, company_number),
        schema=_batch_schema(ids),
        system=BATCH_SYSTEM_PROMPT,
    )
    if result is None:
        reason = getattr(_state, "error", None) or "extraction failed"
        if _batch_failure_is_local(reason):
            return _extract_each_individually(
                usable,
                reports,
                seen_facts,
                company_number=company_number,
                company_name=company_name,
                registry_values=registry_values,
            )
        # A rate limit or an outage will fail the single-page calls just
        # as surely, and retrying twice per page is how a bad minute
        # becomes a request storm.
        for index, _ in usable:
            reports[index].error = reason
        return reports

    for pid in ids:
        reports_by_id[pid].llm_used = True

    _verify_batch_result(
        result,
        reports_by_id,
        pages_by_id,
        seen_facts,
        ids=ids,
        company_number=company_number,
        registry_values=registry_values,
    )

    # Same breadth retry as a single-page read, applied per page. The
    # test uses the facts known before the batch: a directory paired
    # with a productive page should not be retried merely because its
    # neighbour spoke first.
    #
    # A page with no facts and no refusals is just as suspicious: the
    # model may have skipped that block instead of reading it. Re-read
    # it alone when the batch proved it was reading another block.
    # Coverage is worth more than the saved call; a batch that returned
    # nothing at all is left alone, since an empty answer is legitimate
    # and re-reading both pages would give back the whole saving.
    batch_has_facts = any(report.facts for report in reports_by_id.values())
    for pid, (index, page) in zip(ids, usable, strict=True):
        report = reports_by_id[pid]
        restatements = _restatements_only(report.rejected) and (
            page.tier <= TIER_FIRST_PARTY or not seen_before
        )
        silent_skip = not report.rejected and batch_has_facts
        if not report.facts and (restatements or silent_skip):
            replacement = _extract_individually(
                page,
                company_number=company_number,
                company_name=company_name,
                registry_values=registry_values,
                already_accepted=seen_facts,
            )
            if not replacement.retried_for_breadth:
                # The batch was the first pass, so a single-page read is
                # already the second attempt at this page. Record the
                # extra call even though extract_from_page did not make
                # it itself.
                replacement.retried_for_breadth = True
                replacement.retried_rejections = report.rejected
            reports[index] = replacement
            seen_facts.extend(replacement.facts)

    return reports


def _extract_each_individually(
    usable: list[tuple[int, PageInput]],
    reports: list[ExtractionReport],
    seen_facts: list[ExtractedFact],
    *,
    company_number: str,
    company_name: str,
    registry_values: Iterable[str | None],
) -> list[ExtractionReport]:
    for index, page in usable:
        reports[index] = _extract_individually(
            page,
            company_number=company_number,
            company_name=company_name,
            registry_values=registry_values,
            already_accepted=seen_facts,
        )
        seen_facts.extend(reports[index].facts)
    return reports


def _extract_individually(
    page: PageInput,
    *,
    company_number: str,
    company_name: str,
    registry_values: Iterable[str | None],
    already_accepted: Iterable[ExtractedFact],
) -> ExtractionReport:
    return extract_from_page(
        page_text=page.page_text,
        source_url=page.source_url,
        retrieved_at=page.retrieved_at,
        company_number=company_number,
        company_name=company_name,
        tier=page.tier,
        source_type=page.source_type,
        title=page.title,
        registry_values=registry_values,
        already_accepted=already_accepted,
    )


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


# Set per-call because extraction is not re-entrant across threads, and
# threading a mutable through six helpers is worse than a cell the tests
# reset.
#
# The cell has to be thread-local. Companies are extracted concurrently,
# and with one module-level string a model failure on company A could be
# read back by company B, which would then publish A's error against its
# own page. Nothing false reaches a profile that way, but the diagnostic
# stops meaning anything.
_state = threading.local()


def _call_model(
    header: str,
    *,
    schema: dict[str, Any] | None = None,
    system: str | None = None,
):
    """One model call, returning its result or ``None`` on any refusal."""
    # Cleared on entry, so a previous page on this thread cannot supply
    # the reason this one failed.
    _state.error = None
    _state.error_kind = None

    try:
        return llm.complete_json(
            EXTRACTION_SCHEMA if schema is None else schema,
            system=SYSTEM_PROMPT if system is None else system,
            user=header,
        )
    except llm.LLMUnavailable as exc:
        _state.error = str(exc)
        _state.error_kind = "unavailable"
    except llm.LLMRefused as exc:
        _state.error = f"model refused: {exc}"
        _state.error_kind = "refused"
    except llm.LLMTruncated as exc:
        _state.error = f"response truncated: {exc}"
        _state.error_kind = "truncated"
    except llm.LLMCallFailed as exc:
        _state.error = str(exc)
        _state.error_kind = "failed"
    return None


def _batch_failure_is_local(reason: str) -> bool:
    """Whether a failed batch is worth retrying as single pages.

    A truncated response is the batch's own fault: two pages can exceed
    the output budget even when neither page would alone. A malformed
    response is likewise more likely with two blocks to interleave. A
    rate limit or an outage is not local, and retrying it once per page
    turns one failure into several.
    """
    kind = getattr(_state, "error_kind", None)
    if kind in {"refused", "truncated"}:
        return True
    if kind != "failed":
        return False
    lowered = reason.lower()
    return "json" in lowered or "no content" in lowered


def _batch_header(
    ids: list[str],
    pages_by_id: dict[str, PageInput],
    company_name: str,
    company_number: str,
) -> str:
    parts = [f"Company: {company_name} (org.nr {company_number})"]
    for pid in ids:
        page = pages_by_id[pid]
        block = f"=== PAGE {pid} ==="
        if page.title:
            block += f"\nTitle: {page.title}"
        block += f"\nURL: {page.source_url}"
        parts.append(f"{block}\n--- PAGE CONTENT ---\n{page.page_text}")
    return "\n\n".join(parts)


def _batch_schema(page_ids: list[str]) -> dict[str, Any]:
    """The extraction schema plus the page a fact must be attributed to."""
    schema = copy.deepcopy(EXTRACTION_SCHEMA)
    items = schema["properties"]["facts"]["items"]
    items["properties"]["page"] = {
        "type": "string",
        "enum": list(page_ids),
        "description": (
            "Id of the page block whose text contains the quote. "
            "Required: a fact is checked against the page it names."
        ),
    }
    items["required"] = [*items["required"], "page"]
    return schema


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


def _verify_batch_result(
    result,
    reports_by_id: dict[str, ExtractionReport],
    pages_by_id: dict[str, PageInput],
    seen_facts: list,
    *,
    ids: list[str],
    company_number: str,
    registry_values: Iterable[str | None],
) -> None:
    """Check each proposal against the page it claims to quote.

    The model's ``page`` label is a hint, not evidence: a quote located
    in exactly one block is attributed there even if the label is wrong,
    because the text is the thing that was verified. A quote found
    nowhere is refused against the block it named.
    """
    for candidate in (result.data or {}).get("facts") or []:
        if not isinstance(candidate, dict):
            reports_by_id[ids[0]].rejected.append(
                {"candidate": str(candidate)[:200], "reason": "not an object"}
            )
            continue

        quote = str(candidate.get("quote") or "")
        matches = [
            pid
            for pid in ids
            if quote_exists(quote, pages_by_id[pid].page_text)
        ]
        claimed = str(candidate.get("page") or "").strip()

        if claimed in matches:
            pid = claimed
        elif len(matches) == 1 or matches:
            pid = matches[0]
        elif claimed in pages_by_id:
            pid = claimed
        else:
            pid = ids[0]

        page = pages_by_id[pid]
        fact, reason = verify_candidate(
            candidate,
            page_text=page.page_text,
            source_url=page.source_url,
            retrieved_at=page.retrieved_at,
            tier=page.tier,
            source_type=page.source_type,
            company_number=company_number,
            registry_values=registry_values,
            accepted=seen_facts,
        )

        report = reports_by_id[pid]
        if fact is not None:
            report.facts.append(fact)
            seen_facts.append(fact)
        else:
            report.rejected.append(
                {
                    "field": candidate.get("field"),
                    "value": str(candidate.get("value"))[:200],
                    "quote": quote[:300],
                    "reason": reason,
                }
            )
