"""
Page retrieval for extraction.

The agent only ever fetches pages whose identity it has already
established, but that is a reason to be careful rather than careless: a
URL that reached this module came from a search index, so it is
attacker-influenced input.

Two guards matter. Private and link-local addresses are refused so the
agent cannot be pointed at internal infrastructure, and responses are
capped so a large or endless page cannot exhaust memory. HTML is reduced
to text with the standard library alone — the agent needs prose to
reason over, not markup, and an extra parser is not worth the dependency.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse

from src.http import fetch_bytes

REQUEST_TIMEOUT = 20
MAX_BYTES = 2_000_000
MAX_TEXT_CHARS = 200_000

USER_AGENT = "SignalpostAgent/1.0 (+company-intelligence; contact via registry)"

# Elements whose text content is never useful to us.
_SKIP_ELEMENTS = {"script", "style", "noscript", "svg", "template"}

_BLOCK_BREAKS = (
    "p", "div", "br", "li", "tr", "section", "article", "header",
    "footer", "h1", "h2", "h3", "h4", "h5", "h6", "table",
)

_TAG = re.compile(r"<[^>]+>")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_SCRIPT_BLOCK = re.compile(
    r"<(script|style|noscript|svg|template)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_BLOCK_SPLIT = re.compile(
    r"</?(" + "|".join(_BLOCK_BREAKS) + r")\b[^>]*>", re.IGNORECASE
)
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_HREF = re.compile(r"""<a\b[^>]*?\shref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""",
                   re.IGNORECASE)


class UnsafeUrl(ValueError):
    """The URL is malformed, non-HTTP, or resolves to a private address."""


@dataclass
class Page:
    url: str
    title: str
    text: str
    retrieved_at: str
    truncated: bool = False
    links: list[str] = field(default_factory=list)


def _resolve_is_public(host: str) -> bool:
    """True when every address the host resolves to is publicly routable.

    An unresolvable host is treated as unsafe: there is nothing useful
    the agent could fetch from it anyway.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False

    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False

    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def validate_url(url: str) -> str:
    """Return ``url`` if it is safe to fetch, else raise."""
    if not url or not isinstance(url, str):
        raise UnsafeUrl(f"not a URL: {url!r}")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeUrl(f"unsupported scheme: {parsed.scheme}")
    if not parsed.hostname:
        raise UnsafeUrl(f"no host in {url!r}")

    if not _resolve_is_public(parsed.hostname):
        raise UnsafeUrl(
            f"{parsed.hostname} does not resolve to a public address"
        )

    return url


def extract_links(markup: str) -> list[str]:
    """Pull outbound ``href`` targets out of HTML, order preserved.

    Relative hrefs are kept as-is; resolving them needs the page URL,
    which the caller has and this function does not. Duplicates are
    collapsed, since navigation menus repeat the same target many times
    and the duplicates carry no additional signal.
    """
    seen: dict[str, None] = {}

    for match in _HREF.finditer(markup):
        href = match.group(1) or match.group(2) or match.group(3)

        if not href:
            continue

        href = html.unescape(href).strip()

        if not href or href.startswith("#"):
            continue

        seen.setdefault(href, None)

    return list(seen)


def html_to_text(markup: str) -> tuple[str, str]:
    """Reduce HTML to ``(title, text)``.

    Removes script and style bodies entirely, turns block boundaries into
    newlines so sentences do not run together, then collapses runs of
    whitespace.
    """
    title_match = _TITLE.search(markup)
    title = html.unescape(title_match.group(1)).strip() if title_match else ""

    stripped = _SCRIPT_BLOCK.sub(" ", markup)
    stripped = _BLOCK_SPLIT.sub("\n", stripped)
    stripped = _TAG.sub(" ", stripped)

    text = html.unescape(stripped)
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text).strip()

    return title, text


def fetch_page(url: str, max_text_chars: int = MAX_TEXT_CHARS) -> Page:
    """Fetch and flatten one page.

    Raises:
        UnsafeUrl: The URL is malformed or resolves to a private address.
        requests.RequestException: Transport failure after retries.
    """
    safe_url = validate_url(url)

    response = fetch_bytes(
        safe_url,
        timeout=REQUEST_TIMEOUT,
        max_bytes=MAX_BYTES,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
    )

    markup = response.text()
    title, text = html_to_text(markup)
    truncated = response.truncated

    if len(text) > max_text_chars:
        text = text[:max_text_chars]
        truncated = True

    return Page(
        url=safe_url,
        title=title,
        text=text,
        retrieved_at=_iso_now(),
        truncated=truncated,
        links=extract_links(markup),
    )


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "Page",
    "UnsafeUrl",
    "fetch_page",
    "html_to_text",
    "validate_url",
]