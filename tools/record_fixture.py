"""
Records a real run so the end-to-end test can replay it.

Run once, by hand, when the fixture needs refreshing:

    python tools/record_fixture.py

Captures the registry, search and page responses for one company. The
model's answer is deliberately *not* recorded: a replayed extraction has
to quote text that exists in the recorded page, so a canned model
response would either fail verification or force the fixture to embed
assertions about which model happened to run. The test supplies a
deterministic extractor instead and lets quote verification be real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "pipeline.json"

COMPANY = "923609016"

# Trimmed: enough prose for the extractor to quote from, small enough to
# read in a diff.
MAX_TEXT_CHARS = 12_000


def record() -> dict:
    from src import config
    from src.brreg import get_company
    from src.http import fetch_json
    from src.page import fetch_page
    from src.tinyfish_search import build_search_url, search_tinyfish

    config.load_env(force=True)

    pages: dict[str, dict] = {}

    def keep_page(url: str) -> None:
        if url in pages:
            return
        try:
            page = fetch_page(url)
            pages[url] = {
                "kind": "page",
                "status": 200,
                "title": page.title,
                # Flattened prose, not raw HTML. Equinor's first 18 KB of
                # markup is head, navigation and script -- flattening that
                # yields menu labels and no quotable sentence, so the
                # extraction test had nothing real to assert against.
                "text": page.text[:MAX_TEXT_CHARS],
                "links": page.links[:200],
            }
        except Exception as exc:  # noqa: BLE001 - recording is diagnostic
            pages[url] = {"kind": "error", "error": f"{type(exc).__name__}: {exc}"}

    # 1. Registry spine, as the RAW API bodies.
    #
    # Deliberately not get_company()/get_roles()/get_financials(): those
    # return already-normalised dicts, and replaying one in place of the
    # raw response would feed it through normalisation a second time and
    # produce a profile of Nones. The replay must stand in for the
    # network, which is underneath the normalisation.
    registry_urls = {
        "company": (
            f"https://data.brreg.no/enhetsregisteret/api/enheter/{COMPANY}"
        ),
        "roles": (
            f"https://data.brreg.no/enhetsregisteret/api/enheter/{COMPANY}/roller"
        ),
        "financials": (
            f"https://data.brreg.no/regnskapsregisteret/regnskap/{COMPANY}"
        ),
    }
    registry: dict[str, object] = {}
    for key, url in registry_urls.items():
        try:
            registry[key] = fetch_json(url, validate=False)
        except Exception as exc:  # noqa: BLE001
            registry[key] = {"error": f"{type(exc).__name__}: {exc}"}

    # The normalised company record supplies the query and the official
    # site; recorded separately so a replay does not have to re-derive it.
    try:
        normalised = get_company(COMPANY)
        company_view = {
            "company_name": normalised.get("company_name"),
            "website": normalised.get("website"),
        }
        query = f"{normalised.get('company_name')} {COMPANY}"
    except Exception:  # noqa: BLE001
        company_view = {"company_name": None, "website": None}
        query = COMPANY

    # 2. Search discovery.
    results = search_tinyfish(query, max_results=8, use_cache=False)

    # 3. The pages the crawler would reach.
    for result in results[:4]:
        keep_page(result["url"])

    official = company_view.get("website")
    if official:
        home = f"https://{official.lstrip('/')}"
        keep_page(home)
        # Second hop, to give the fixture a real same-domain follow-up.
        keep_page(f"{home}/about-us")

    return {
        "_comment": (
            "Recorded HTTP responses for the end-to-end test. Registry "
            "entries are raw API bodies. Regenerate with: "
            "python tools/record_fixture.py"
        ),
        "company_number": COMPANY,
        "company_view": company_view,
        "query": query,
        "registry": registry,
        "search": results,
        "search_url": build_search_url(query),
        "pages": pages,
    }


def main() -> int:
    data = record()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    size = FIXTURE.stat().st_size
    print(f"wrote {FIXTURE} ({size / 1024:.0f} KB)")
    print(f"  registry keys : {sorted(data['registry'])}")
    print(f"  search hits   : {len(data['search'])}")
    print(f"  pages         : {len(data['pages'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
