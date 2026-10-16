"""
Build a company list for a scale test.

Real organizations, not invented ones. The first version generated
candidate numbers by walking a serial range and recomputing the MOD-11
check digit, then probing each one. That produced 300 requests and zero
hits: the arithmetic was correct, but companies are not dense enough at
stride 37 for random sampling to find them. Paginating the registry's own
list endpoint returns real entities directly.

Companies with a registered website are preferred, because a batch
without them has no first-party page for the crawler and the enrichment
path is never exercised.

    python tools/build_company_list.py --count 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.http import fetch_json  # noqa: E402
from src.orgnumber import InvalidCompanyNumber, validate  # noqa: E402

LIST_URL = "https://data.brreg.no/enhetsregisteret/api/enheter"


def collect(*, wanted: int, max_pages: int, size: int) -> list[str]:
    """Page through Enhetsregisteret, preferring entities with websites."""
    with_site: list[str] = []
    without_site: list[str] = []
    seen: set[str] = set()

    for page in range(max_pages):
        url = f"{LIST_URL}?size={size}&page={page}"

        # Enhetsregisteret caps the result window, so a deep page comes
        # back 400 rather than empty. Stopping there is correct; letting
        # it raise would discard every company already collected.
        try:
            data = fetch_json(url, validate=False)
        except Exception as exc:  # noqa: BLE001 - end of the window
            print(f"  pagination stopped at page {page}: {type(exc).__name__}")
            break

        entities = (data or {}).get("_embedded", {}).get("enheter", [])
        if not entities:
            break

        for entity in entities:
            number = entity.get("organisasjonsnummer")
            if not number or number in seen:
                continue
            seen.add(number)

            try:
                validate(number)
            except InvalidCompanyNumber:
                continue

            # Delisted shells make poor test subjects: nothing to say and
            # often no website to crawl. The list endpoint leaves
            # naeringskode empty and carries industry in `aktivitet`.
            if not entity.get("aktivitet"):
                continue

            bucket = (
                with_site if entity.get("hjemmeside") else without_site
            )
            bucket.append(number)

        if len(with_site) >= wanted:
            break

    ordered = with_site[:wanted]
    if len(ordered) < wanted:
        ordered += without_site[: wanted - len(ordered)]

    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(prog="build_company_list.py")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "company_numbers_50.txt"),
    )
    parser.add_argument("--max-pages", type=int, default=12)
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()

    numbers = collect(
        wanted=args.count, max_pages=args.max_pages, size=args.size
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(numbers) + "\n", encoding="utf-8")

    print(f"wrote {out_path} with {len(numbers)} organizations")
    if len(numbers) < args.count:
        print(
            f"  only {len(numbers)} found; raise --max-pages",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
