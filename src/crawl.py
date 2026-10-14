"""
Bounded same-domain crawl.

A company's registered website is already the strongest source the
agent can find: same domain as the registry record, so identity is
established before a single word is read. One homepage, however, is a
thin basis for a profile -- and the deepest, cheapest facts on most
company sites sit on the About and Services pages, not the front door.

Two properties make this safe where arbitrary web search is not:

* Every URL is on the registered domain, so it inherits the identity
  verification the homepage already earned. The crawler never widens
  the trust boundary; it only walks inside one that has been proven.
* The walk is depth 1 and hard-capped, so it cannot run away into a
  sitemap.

Link choice is biased toward *stable* pages. A homepage's most-linked
pages are frequently press releases and recruitment ads, which produce
verifiable facts that expire within a month. Path scoring prefers what
describes the business rather than what announces an event, while still
following ordinary navigation rather than enforcing an allowlist --
allowlists silently drop every company whose site does not match the
guess, and these sites are not built to a template.
"""

from __future__ import annotations

import re
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

from src.sources import get_domain

# Paths that describe the business rather than announce an event.
# Ordered by weight; the first match wins.
_POSITIVE_HINTS: tuple[tuple[str, int], ...] = (
    (r"om-oss|about|who-we-are|profile", 40),
    (r"tjenester|services|what-we-do|virksomhet|products", 35),
    (r"kontakt|contact", 30),
    (r"organisasjon|historie|history", 25),
    (r"karriere|careers|jobb", 15),
)

# Checked first, and deliberately so. These paths share vocabulary with
# the positives -- /about-us/accessibility-statement contains "about" --
# but describe nothing about the business. Evaluating positives first
# would let every one of them win on the substring it happens to share.
_NEGATIVE_HINTS: tuple[tuple[str, int], ...] = (
    (r"press|news|nyheter|media|blogg", -50),
    (r"personale|ansatte|ledelse|team", -60),
    (r"personvern|privacy|cookie|tilgjengelighet|accessibility|"
     r"vilk(a|å)r|terms|ansvarserkl(a|æ)ring|erkl(a|æ)ring|"
     r"personvernpolitikk|sitemap|s(u|ø)k", -70),
)

_POSITIVE_PATTERNS = tuple(
    (re.compile(p, re.IGNORECASE), w) for p, w in _POSITIVE_HINTS
)
_NEGATIVE_PATTERNS = tuple(
    (re.compile(p, re.IGNORECASE), w) for p, w in _NEGATIVE_HINTS
)

# File types that are never worth a model call.
_BINARY_HINTS = re.compile(
    r"\.(pdf|docx?|xlsx?|pptx?|zip|gz|tar|rar|7z|mp3|mp4|avi|mov|"
    r"webm|woff2?|ttf|eot|ico|jpg|jpeg|png|gif|svg|webp|css|js|json|"
    r"rss|atom)($|\?)",
    re.IGNORECASE,
)

_NON_PAGE_SCHEMES = ("mailto:", "tel:", "sms:", "javascript:", "data:", "#")

# Depth one only. Deeper walks buy recency (news archives, tag pages)
# rather than description.
MAX_DEPTH = 1

DEFAULT_MAX_CRAWL_LINKS = 5


# Every followed page costs a request and a model call, so a negative
# score means "do not read this", not merely "read it last". A site
# whose only internal links are press pages yields no follow-ups, which
# is correct: its homepage was already read, and its press archive adds
# facts that expire within a month.
MIN_FOLLOWUP_SCORE = 0


def _path_weight(path: str) -> int:
    """Score a path by how likely it is to describe the business."""
    for pattern, weight in _NEGATIVE_PATTERNS:
        if pattern.search(path):
            return weight

    for pattern, weight in _POSITIVE_PATTERNS:
        if pattern.search(path):
            return weight

    return 10


def _depth_penalty(path: str) -> int:
    """Prefer shallow pages.

    A deep path is usually an archive, a tag page, or a single press
    item reached through a date-stamped tree. Those describe an event
    rather than the business, and they are the paths that go stale.
    """
    segments = [s for s in path.split("/") if s]
    return max(0, len(segments) - 2) * 8


def _looks_like_page(href: str) -> bool:
    if not href:
        return False

    lowered = href.strip().lower()

    if lowered.startswith(_NON_PAGE_SCHEMES):
        return False

    return not _BINARY_HINTS.search(lowered)


def _normalise(base_url: str, href: str) -> str | None:
    """Resolve ``href`` against ``base_url`` and drop it if unusable."""
    try:
        absolute = urljoin(base_url, href.strip())
    except ValueError:
        return None

    absolute, _fragment = urldefrag(absolute)

    if not absolute.startswith(("http://", "https://")):
        return None

    # "https://example.no" and "https://example.no/" are the same page.
    # Left distinct, a site linking to its own root would be re-read as a
    # follow-up, spending a model call to fetch the homepage twice.
    parsed = urlparse(absolute)
    if parsed.path == "/" and not parsed.query:
        absolute = urlunparse(parsed._replace(path=""))

    return absolute


def plan_crawl(
    base_url: str,
    links: list[str],
    registered_domain: str,
    *,
    max_links: int = DEFAULT_MAX_CRAWL_LINKS,
    already_seen: set[str] | None = None,
) -> list[str]:
    """Choose which same-domain pages to read next.

    Args:
        base_url: The page the links were found on, used to resolve
            relative hrefs.
        links: Raw ``href`` values from that page.
        registered_domain: Domain the registry recorded. Anything
            outside it is discarded outright.
        max_links: Hard ceiling on returned URLs.
        already_seen: URLs already fetched or queued.

    Returns:
        Up to ``max_links`` absolute URLs on the registered domain,
        ordered by descending preference, stable-path pages first.
        Pages that scored negative -- press, personnel, compliance --
        are dropped rather than ranked last.
    """
    if not links or max_links <= 0:
        return []

    seen = set(already_seen or set())
    seen.add(_normalise(base_url, base_url) or base_url)

    scored: list[tuple[int, str]] = []

    for href in links:
        if not _looks_like_page(href):
            continue

        absolute = _normalise(base_url, href)

        if absolute is None:
            continue

        if absolute in seen:
            continue

        # The trust boundary. Same registrable domain as the registry
        # record, or nothing.
        if get_domain(absolute) != registered_domain:
            continue

        path = urlparse(absolute).path
        score = _path_weight(path) - _depth_penalty(path)

        if score < MIN_FOLLOWUP_SCORE:
            continue

        scored.append((score, absolute))
        seen.add(absolute)

    scored.sort(key=lambda item: (-item[0], item[1]))

    return [url for _weight, url in scored[:max_links]]


__all__ = [
    "DEFAULT_MAX_CRAWL_LINKS",
    "MAX_DEPTH",
    "plan_crawl",
]
