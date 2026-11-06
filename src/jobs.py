"""
Careers-page reading.

Job openings are the one family that lives on a page nobody links for
its content: a footer "careers" link, an ATS subdomain, a Norwegian
``/ledige-stillinger``. The general extractor reads such a page like any
other and mostly produces noise -- deadlines and location names without
a title -- so this module reads it structurally instead.

Two signals are trusted, both of them things the page states outright:

* A careers URL that was actually fetched (proof the page exists).
* Individual job-detail links on that page, counted and titled from
  their slugs. A link to ``/stillinger/1234-senior-ingenior`` is the
  publisher's own claim that the position is open.

No model is involved, so a result is reproducible from the snapshot.
When a careers page was read and carries neither signal, the caller
records ``jobs: not_available`` rather than leaving the field absent --
"checked, nothing open" and "never looked" are different answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

# Paths that are a careers landing page for this company, in Norwegian
# and English. Matched against the path only, so a query string cannot
# smuggle a false positive in.
CAREERS_PATH = re.compile(
    r"/(careers?|jobs?|jobber|karriere|karriär|jobb|stillinger|"
    r"ledige-stillinger|arbeid-hos-oss|work-with-us|join-us|bli-med)"
    r"(/|$)",
    re.IGNORECASE,
)

# A link to one posting rather than to the careers landing page.
JOB_LINK = re.compile(
    r"(/stilling(er)?/|/jobs?/|/jobber/|/vacanc(y|ies)/|"
    r"/positions?/|/apply/|/ad/\d+|jobid=|job_id=|/job/\d+)",
    re.IGNORECASE,
)

# Applicant-tracking systems host the individual postings, so any link
# to one of these on a careers page is a posting link regardless of path.
ATS_HOSTS = (
    "jobbnorge.no",
    "finn.no",
    "teamtailor.com",
    "recruitee.com",
    "myworkdayjobs.com",
    "easycruit.com",
    "reachmee.com",
    "hr-manager.net",
    "rekrutteringsbistand.no",
    "webcruiter.no",
    "lever.co",
    "greenhouse.io",
    "bamboohr.com",
    "personio.",
    "oraclecloud.com",
)

# Text that announces a listing section. Used only to title a page that
# has posting links but no readable slugs; never to invent an opening.
HIRING_MARKERS = re.compile(
    r"(ledige stillinger|vi s(ø|o)ker|vi ser etter|s(ø|o)k stilling|"
    r"se ledige|open positions|open roles|we are hiring|join our team|"
    r"current vacancies|view jobs|apply now|arbeid hos oss|bli med)",
    re.IGNORECASE,
)

_MAX_TITLES = 12

# Slug words that title a page rather than a position. A link to
# "Subscription.aspx" on an ATS is a posting link by host, but the
# resulting "title" would claim the company is hiring a Subscription.
_NOISE_TITLES = frozenset(
    {
        "apply",
        "careercenter",
        "career-center",
        "index",
        "login",
        "log-in",
        "overview",
        "register",
        "search",
        "signin",
        "sign-in",
        "subscription",
    }
)

# Listing indexes ("/careers/vacancies", "/jobb") link to *more* careers
# pages, not to one position. Read as postings they publish a title like
# "Vacancies", which is true of the URL and false of the business.
_INDEX_SEGMENT = re.compile(
    r"^(careers?|karriere|karri(ä|a)r|jobs?|jobber|jobb|stillinger|"
    r"ledige-stillinger|vacancies|vacancy|positions?|open-positions|"
    r"open-roles|arbeid-hos-oss|work-with-us|join-us|bli-med)$",
    re.IGNORECASE,
)


def is_careers_url(url: str) -> bool:
    """Whether ``url`` is a careers landing page."""
    try:
        path = urlparse(url).path or ""
    except ValueError:
        return False
    return bool(CAREERS_PATH.search(path or "/"))


def _is_job_link(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if any(ats in host for ats in ATS_HOSTS):
        return True
    return bool(JOB_LINK.search(parsed.path or ""))


def _last_segment(url: str) -> str:
    try:
        path = urlparse(url).path or ""
    except ValueError:
        return ""
    segments = [segment for segment in path.split("/") if segment]
    return segments[-1] if segments else ""


def is_careers_index(url: str) -> bool:
    """Whether ``url`` is a careers listing rather than one opening."""
    return bool(_INDEX_SEGMENT.match(_last_segment(url))) and is_careers_url(url)


def careers_index_links(
    url: str,
    links: list[str] | None = None,
    *,
    max_links: int = 2,
    has_postings: bool = False,
) -> list[str]:
    """Careers-listing links found on a careers page, in page order.

    A landing page that lists no openings often links the page that
    does ("/careers/vacancies"). Reading it costs one request and turns
    an empty jobs family into titles; the caller must still verify the
    target page's identity before publishing anything it says.

    A page that already lists openings hands over nothing: its index
    links would only lead to other listings, and following them would
    spend the walk's budget without adding a title.
    """
    found: list[str] = []
    if has_postings:
        return found

    for href in links or []:
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        try:
            absolute = urljoin(url, href.strip())
        except ValueError:
            continue
        if not absolute.startswith(("http://", "https://")):
            continue
        if absolute.rstrip("/") == url.rstrip("/"):
            continue
        if is_careers_index(absolute) and absolute not in found:
            found.append(absolute)
        if len(found) >= max_links:
            break

    return found


def _title_from_url(url: str) -> str | None:
    """Turn a posting slug into a human title, or ``None`` if unusable."""
    try:
        path = urlparse(url).path
    except ValueError:
        return None
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return None
    slug = segments[-1]
    slug = re.split(r"\.(?:html?|php|aspx?)$", slug, flags=re.IGNORECASE)[0]
    # Drop a leading numeric id: "1234-senior-ingenior".
    slug = re.sub(r"^\d{2,}[-_]+", "", slug)
    if slug.lower() in _NOISE_TITLES:
        return None
    words = [part for part in re.split(r"[-_+]+", slug) if part.isalpha()]
    if not words:
        return None
    if len(words) == 1:
        # Norwegian compounds carry the whole title: "prosjektleder".
        if len(words[0]) < 6:
            return None
        title = words[0].title()
        return title if len(title) <= 80 else None
    title = " ".join(words)
    if title.islower():
        title = title.title()
    if not 4 <= len(title) <= 80:
        return None
    if title.lower() in _NOISE_TITLES:
        return None
    return title


@dataclass
class JobFindings:
    """What a fetched careers page proves."""

    careers_url: str
    titles: list[str] = field(default_factory=list)
    posting_links: int = 0
    snippet: str = ""

    @property
    def checked(self) -> bool:
        return True


def read_careers_page(
    text: str,
    url: str,
    links: list[str] | None = None,
) -> JobFindings:
    """Read one fetched careers page without a model.

    Args:
        text: Visible page text.
        url: The careers page URL that was fetched.
        links: Raw hrefs found on the page.

    Returns:
        The page's posting links and the titles recoverable from them.
        ``titles`` is empty when the page was read but lists nothing.
    """
    resolved: list[str] = []
    for href in links or []:
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        try:
            absolute = urljoin(url, href.strip())
        except ValueError:
            continue
        if not absolute.startswith(("http://", "https://")):
            continue
        if absolute.rstrip("/") == url.rstrip("/"):
            continue
        if _is_job_link(absolute) and not is_careers_index(absolute):
            resolved.append(absolute)

    # Posting links repeat in navigation; one title per unique slug.
    unique = list(dict.fromkeys(resolved))
    titles: list[str] = []
    for link in unique:
        title = _title_from_url(link)
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= _MAX_TITLES:
            break

    snippet = ""
    marker = HIRING_MARKERS.search(text or "")
    if marker:
        start = max(0, marker.start() - 40)
        snippet = (text or "")[start : marker.end() + 80].strip()

    return JobFindings(
        careers_url=url,
        titles=titles,
        posting_links=len(unique),
        snippet=snippet,
    )


__all__ = [
    "JobFindings",
    "careers_index_links",
    "is_careers_index",
    "is_careers_url",
    "read_careers_page",
]
