"""
Registered-website verification.

Brreg states a website, but that website is not necessarily the target
company's own site: it is often the parent group's, a brand site or a
holding company's. A site is first-party only when it carries the target
organization number or the exact registered company name. Social links
are published only from a site that passed that check, and only when the
account handle matches the company too.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import urljoin, urlparse

from src.identity import CompanyIdentity, strip_legal_form
from src.page import Page
from src.social import account_from_url, social_handle_matches
from src.sources import org_numbers_in
from src.text import fold


def _compact(value: str) -> str:
    return "".join(ch for ch in fold(value) if ch.isalnum())


@dataclass
class WebsiteCheck:
    """The result of testing one page against a company identity."""

    url: str
    verified: bool
    basis: str
    org_number_found: bool
    exact_name_found: bool
    previous_name_found: bool = False
    checked_at: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def verify_registered_page(
    identity: CompanyIdentity,
    page: Page,
) -> WebsiteCheck:
    """Decide whether ``page`` is first-party evidence for ``identity``.

    The organization number outranks the name. Company names can be
    reused or shared by a group; the number cannot.
    """
    text = f"{page.title}\n{page.text}"
    numbers = org_numbers_in(text)
    org_number_found = identity.company_number in numbers

    compact_text = _compact(text)
    current_name = _compact(strip_legal_form(identity.name))
    exact_name_found = (
        len(current_name) >= 4 and current_name in compact_text
    )

    previous_name_found = False
    if not exact_name_found:
        for name in identity.previous_names:
            compact_name = _compact(strip_legal_form(name or ""))
            if len(compact_name) >= 4 and compact_name in compact_text:
                previous_name_found = True
                break

    if org_number_found:
        basis = "organization_number"
    elif exact_name_found:
        basis = "exact_name"
    elif previous_name_found:
        basis = "previous_name"
    else:
        basis = "not_found"

    return WebsiteCheck(
        url=page.url,
        verified=org_number_found or exact_name_found or previous_name_found,
        basis=basis,
        org_number_found=org_number_found,
        exact_name_found=exact_name_found,
        previous_name_found=previous_name_found,
        checked_at=page.retrieved_at,
    )


def collect_social_links(
    identity: CompanyIdentity,
    page: Page,
) -> list[dict]:
    """Validated social accounts linked from a first-party page."""
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for href in page.links:
        url = urljoin(page.url, href)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            continue
        account = account_from_url(url)
        if account is None:
            continue
        if not social_handle_matches(
            url,
            company_name=identity.name,
            previous_names=identity.previous_names,
        ):
            continue
        key = (account[0], account[1].lower())
        if key in seen:
            continue
        seen.add(key)
        found.append(
            {
                "platform": account[0],
                "handle": account[1],
                "url": url,
                # The verified page the account was linked from. Without
                # it the handle is an assertion with no provenance and
                # cannot be published as a claim.
                "source_url": page.url,
                "retrieved_at": page.retrieved_at,
                # The link's own label, when it has one. It is what the
                # claim quotes, because it is on the page.
                "anchor_text": page.anchor_text.get(href, ""),
            }
        )

    return found
