"""
News mentions from the Google News RSS search feed.

Keyless and free, which is why it is the first external-intelligence
source: a search provider key is not available in a keyless run, but the
RSS search endpoint needs only a User-Agent. The feed is a search result,
not a curated record, so a hit is published only when the company name
appears in the item itself -- the query that found it is not evidence
that it is about the company.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

from src.identity import strip_legal_form
from src.text import fold

DEFAULT_ENDPOINT = "https://news.google.com/rss/search"

# Google answers a bare client with a consent/robot page, and the feed
# is not served without one.
USER_AGENT = (
    "SignalpostAgent/1.0 (+company-intelligence; contact via registry)"
)

# Bound the window so a company that made news once in 2015 is not
# presented as current activity.
QUERY_WINDOW = "when:1y"

MAX_ITEMS = 2

# A three-letter name matches too much unrelated text to be evidence.
MIN_KEY_LENGTH = 4


@dataclass(frozen=True)
class NewsItem:
    title: str
    publisher: str
    published: str
    link: str
    summary: str


def feed_url(
    company_name: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    window: str = QUERY_WINDOW,
) -> str:
    """The search feed URL for one company name."""
    query = f'"{strip_legal_form(company_name)}"'
    if window:
        query = f"{query} {window}"
    return (
        f"{endpoint}?q={quote(query)}&hl=nb&gl=NO&ceid=NO:no"
    )


def _published_date(value: str) -> str | None:
    """The RFC-822 ``pubDate`` as ``yyyy-mm-dd`` in UTC."""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        moment = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).date().isoformat()


def _headline(title: str, publisher: str) -> str:
    """Drop the `` - Publisher`` suffix Google appends to every title."""
    text = " ".join((title or "").split())
    suffix = f" - {publisher}".strip()
    if publisher and text.endswith(suffix):
        text = text[: -len(suffix)].strip()
    return text


def parse_feed(body: bytes) -> list[NewsItem]:
    """Parse an RSS body into items, newest first."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []

    items: list[NewsItem] = []
    for node in root.findall("./channel/item"):
        title = node.findtext("title") or ""
        link = node.findtext("link") or ""
        if not title or not link:
            continue
        publisher = node.findtext("source") or ""
        items.append(
            NewsItem(
                title=_headline(title, publisher),
                publisher=" ".join(publisher.split()),
                published=_published_date(node.findtext("pubDate") or "")
                or "",
                link=link.strip(),
                summary=re.sub(
                    r"<[^>]+>", " ", node.findtext("description") or ""
                ),
            )
        )

    items.sort(key=lambda item: item.published, reverse=True)
    return items


def _normalize(value: str) -> str:
    """Lowercase, transliterated, punctuation as single spaces."""
    return " ".join(
        "".join(
            character if character.isalnum() else " " for character in fold(value)
        ).split()
    )


def _key(value: str) -> str:
    """The matchable form of a name: normalized, legal form removed."""
    return _normalize(strip_legal_form(value or ""))


def key_length(value: str) -> int:
    """Characters in the name once separators are dropped."""
    return len(_key(value).replace(" ", ""))


def _mentions(key: str, text: str) -> bool:
    """True when ``key`` appears in ``text`` as a whole phrase."""
    if not key:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])"
    return re.search(pattern, _normalize(text)) is not None


def match_news(
    items: list[NewsItem],
    *,
    company_name: str,
    previous_names: tuple[str, ...] | list[str] = (),
    limit: int = MAX_ITEMS,
) -> list[NewsItem]:
    """Items that name this company, newest first, capped at ``limit``.

    Both the headline and the feed summary must be considered: Google
    matches on the description as well as the title, so the name is
    often only in the summary.
    """
    keys = [
        key
        for key in (
            _key(company_name),
            *(_key(name) for name in previous_names),
        )
        if key_length(key) >= MIN_KEY_LENGTH
    ]
    if not keys:
        return []

    matches = [
        item
        for item in items
        if any(_mentions(key, f"{item.title} {item.summary}") for key in keys)
    ]
    return matches[:limit]


def search_news(
    company_name: str,
    *,
    fetch,
    endpoint: str = DEFAULT_ENDPOINT,
    window: str = QUERY_WINDOW,
) -> list[NewsItem]:
    """Fetch and parse the feed for ``company_name``.

    ``fetch`` takes a URL and returns the body bytes, so the caller owns
    request counting, pacing and the address guard.
    """
    body = fetch(feed_url(company_name, endpoint=endpoint, window=window))
    return parse_feed(body or b"")


def news_facts(
    matches: list[NewsItem],
    *,
    retrieved_at: str,
) -> list[dict]:
    """Fact records for matched items, newest first.

    The claim is the headline; the evidence is the same headline, which
    is the text the cited article is published under.

    Items without a parseable publication date are dropped: a real news
    post carries one, and an undated or feed-modified timestamp is not a
    publish date. If the date cannot be read, the item is not published.
    """
    facts: list[dict] = []
    for item in matches:
        if not item.published:
            continue
        facts.append(
            {
                "field": "news",
                "value": item.title,
                "source_url": item.link,
                "retrieved_at": retrieved_at,
                "evidence_text": item.title,
                "confidence": 0.6,
                "source_tier": 5,
                "source_type": "news_source",
                "as_of": item.published or None,
            }
        )
    return facts
