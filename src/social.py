"""
Social account checks.

A social URL found in a page footer is not evidence that the account
belongs to the target company: footers inherit the parent group's, the
web agency's or the theme author's links. This module extracts only an
account handle from a URL and accepts it when that handle matches the
company name, a previous name or the registered domain label.
"""

from __future__ import annotations

from urllib.parse import urlparse

from src.identity import strip_legal_form
from src.sources import get_domain
from src.text import fold

_SOCIAL_DOMAINS = {
    "facebook.com": "facebook",
    "fb.com": "facebook",
    "linkedin.com": "linkedin",
    "instagram.com": "instagram",
    "x.com": "x",
    "twitter.com": "twitter",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "tiktok.com": "tiktok",
}

_NON_COMPANY_FACEBOOK_PATHS = {
    "groups",
    "people",
    "profile.php",
    "events",
    "watch",
    "marketplace",
    "gaming",
}

_NON_COMPANY_LINKEDIN_PATHS = {"in", "pulse", "feed", "posts", "school"}

_NON_COMPANY_INSTAGRAM_PATHS = {
    "p",
    "reel",
    "reels",
    "explore",
    "stories",
    "accounts",
    "tv",
}

_NON_COMPANY_X_PATHS = {
    "i",
    "intent",
    "share",
    "home",
    "search",
    "hashtag",
    "messages",
    "notifications",
    "settings",
}

_LEGAL_SUFFIXES = (
    "limited",
    "gmbh",
    "sarl",
    "corp",
    "plc",
    "asa",
    "oyj",
    "ab",
    "as",
    "sa",
    "sas",
    "bv",
    "nv",
    "ltd",
    "llc",
    "inc",
    "oy",
    "ag",
)


def _compact(value: str) -> str:
    return "".join(ch for ch in fold(value) if ch.isalnum())


def _domain_platform(domain: str) -> str | None:
    for known, platform in _SOCIAL_DOMAINS.items():
        if domain == known or domain.endswith("." + known):
            return platform
    return None


def social_platform(url: str) -> str | None:
    """The social platform for ``url``, or ``None`` for any other URL."""
    return _domain_platform(get_domain(url))


def is_social_url(url: str) -> bool:
    """True when ``url`` is on a known social platform."""
    return social_platform(url) is not None


def account_from_url(url: str) -> tuple[str, str] | None:
    """Return ``(platform, handle)`` only for a company-page URL.

    Group, personal-profile, post and generic paths return ``None`` even
    though the domain is social, because their handle does not identify
    a company account.
    """
    platform = social_platform(url)
    if platform is None:
        return None

    parsed = urlparse(url if "://" in url else f"https://{url}")
    segments = [segment for segment in parsed.path.split("/") if segment]
    if not segments:
        return None

    first = segments[0].lower()

    if platform == "facebook":
        if first in _NON_COMPANY_FACEBOOK_PATHS:
            return None
        if first == "pages" and len(segments) >= 2:
            return platform, segments[1]
        return platform, segments[0]

    if platform == "linkedin":
        if first in _NON_COMPANY_LINKEDIN_PATHS:
            return None
        if first in ("company", "showcase") and len(segments) >= 2:
            return platform, segments[1]
        return None

    if platform == "instagram":
        if first in _NON_COMPANY_INSTAGRAM_PATHS:
            return None
        return platform, segments[0]

    if platform in ("x", "twitter"):
        if first in _NON_COMPANY_X_PATHS:
            return None
        if len(segments) >= 2 and segments[1].lower() in (
            "status",
            "statuses",
        ):
            return None
        return platform, segments[0]

    if platform == "youtube":
        if first.startswith("@"):
            return platform, first[1:]
        if first in ("c", "user") and len(segments) >= 2:
            return platform, segments[1]
        return None

    if platform == "tiktok":
        if first == "video":
            return None
        handle = first[1:] if first.startswith("@") else ""
        return (platform, handle) if handle else None

    return None


def social_handle_matches(
    url: str,
    *,
    company_name: str,
    previous_names: tuple[str, ...] | list[str] = (),
    registered_website: str | None = None,
) -> bool:
    """True when the social handle visibly names this company.

    Exact handle equality is required after removing the legal form and
    punctuation. A parent-group handle such as ``acmegroup`` does not
    match ``Acme Norge AS`` merely because they share a token.
    """
    account = account_from_url(url)
    if account is None:
        return False

    _, handle = account
    normalized = _compact(handle)
    if len(normalized) < 3:
        return False

    names = [company_name, *previous_names]
    candidates = {
        compact
        for name in names
        if (compact := _compact(strip_legal_form(name or "")))
        and len(compact) >= 3
    }

    domain = get_domain(registered_website or "")
    if domain:
        label = domain.split(".")[0]
        compact_label = _compact(label)
        if len(compact_label) >= 3:
            candidates.add(compact_label)
        else:
            candidates.add(_compact(domain.replace(".", "")))

    if normalized in candidates:
        return True

    for candidate in candidates:
        if len(candidate) < 4:
            continue
        for suffix in _LEGAL_SUFFIXES:
            if normalized == candidate + suffix:
                return True
    return False
