"""
URL safety, shared by everything that makes an outbound request.

Lives apart from ``src/http.py`` and ``src/page.py`` because all three
need it: ``http`` must validate every redirect hop, ``page`` validates
before fetching, and neither can import the other without a cycle.

The guard answers one question -- may the agent open a connection to
this host? -- and the answer must be re-asked at every hop rather than
only for the URL the caller supplied.

That distinction is the whole point. A discovered URL that 302s to
``http://169.254.169.254/latest/meta-data/`` passes the check on the
first URL and reaches cloud instance metadata on the second. Delegating
redirects to the HTTP library means nothing re-validates the target, so
the guard applies to the URL the agent decided to check and not to the
one it ends up at.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# Few legitimate pages redirect more than this. A search index pointing at
# a company website should reach it in one or two hops; a chain longer
# than this is either a mistake or a way to burn the request budget.
DEFAULT_MAX_REDIRECTS = 5

ALLOWED_SCHEMES = ("http", "https")


class UnsafeUrl(ValueError):
    """The URL is malformed, non-HTTP, or resolves to a private address."""


class TooManyRedirects(Exception):
    """The redirect chain exceeded its cap."""


def resolve_is_public(host: str) -> bool:
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

    url = url.strip()

    # The scheme is inspected *before* any coercion. Prepending "https://"
    # first turned "file:///etc/passwd" into
    # "https://file:///etc/passwd", whose scheme then read as https and
    # passed -- so the scheme check below was unreachable, and rejection
    # only happened by accident when the mangled host failed to resolve.
    if "://" in url:
        if urlparse(url).scheme.lower() not in ALLOWED_SCHEMES:
            raise UnsafeUrl(f"unsupported scheme: {urlparse(url).scheme}")
    else:
        # A bare host such as "example.no/about" is assumed HTTPS rather
        # than HTTP: the agent fetches public pages, and silently
        # downgrading a link is worse than failing it.
        url = "https://" + url

    parsed = urlparse(url)

    if not parsed.hostname:
        raise UnsafeUrl(f"no host in {url!r}")

    # Credentials in a URL are a phishing pattern and would be written to
    # the cache directory verbatim.
    if parsed.username or parsed.password:
        raise UnsafeUrl("credentials in URL")

    if not resolve_is_public(parsed.hostname):
        raise UnsafeUrl(f"{parsed.hostname} does not resolve to a public address")

    return url


def redirect_target(response, *, max_redirects: int) -> str | None:
    """The next URL in a redirect chain, or ``None`` when finished.

    Raises:
        TooManyRedirects: The chain is longer than ``max_redirects``.
    """
    status = response.status_code

    if status not in (301, 302, 303, 307, 308):
        return None

    location = response.headers.get("Location")
    if not location:
        # A redirect with no target is a dead end, not a loop. Returning
        # None lets the caller read the body and decide.
        return None

    if max_redirects <= 0:
        raise TooManyRedirects(
            f"redirect chain exceeded {max_redirects} hops while following"
        )

    return location


__all__ = [
    "ALLOWED_SCHEMES",
    "DEFAULT_MAX_REDIRECTS",
    "TooManyRedirects",
    "UnsafeUrl",
    "redirect_target",
    "resolve_is_public",
    "validate_url",
]
