"""
Web search through TinyFish.

Search is billable per call and the API authenticates with a header, not
a URL parameter. Two consequences shape this module:

* The cache is keyed on the *expanded* URL, query string included.
  ``cached_json`` keys on the URL alone, so passing the bare endpoint
  would make every query share one entry and hand each company another
  company's results.
* Credentials travel in the ``X-API-Key`` header, which ``cached_json``
  forwards on a miss and never writes to disk. Putting the key in the
  URL would leak it into the cache directory.

``max_results`` is deliberately excluded from the cache key: it trims a
result set that was already fetched, so two calls differing only in
limit reuse one network request.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from src.cache import SEARCH_TTL_SECONDS, cached_json
from src.config import load_env

# Reads the repo's .env from the working directory outward, rather than
# searching upward from this file -- which found nothing once the package
# was installed.
load_env()

TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"

SEARCH_PARAMS = {"location": "NO", "language": "no"}


def build_search_url(query: str) -> str:
    """The exact URL that will be requested, query string included.

    Used both as the request target and as the cache key, so the two can
    never drift apart.
    """
    params = {"query": query, **SEARCH_PARAMS}
    return requests.Request("GET", TINYFISH_SEARCH_URL, params=params).prepare().url


def _normalise(results: list[dict[str, Any]], max_results: int) -> list[dict[str, str]]:
    normalised: list[dict[str, str]] = []

    for result in results[:max_results]:
        url = result.get("url")

        if not url:
            continue

        normalised.append(
            {
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("snippet", ""),
            }
        )

    return normalised


def search_tinyfish(
    query: str,
    max_results: int = 10,
    *,
    use_cache: bool = True,
) -> list[dict]:
    api_key = os.getenv("TINYFISH_API_KEY")

    if not api_key:
        raise RuntimeError("TINYFISH_API_KEY is not configured")

    url = build_search_url(query)
    headers = {"X-API-Key": api_key}

    if use_cache:
        data, _meta = cached_json(
            url, headers=headers, ttl_seconds=SEARCH_TTL_SECONDS
        )
    else:
        from src.http import fetch_json

        # fetch_json records the request itself, so no extra record()
        # here or the tracker counts this call twice.
        data = fetch_json(url, headers=headers)

    if not data:
        return []

    results = data.get("results", []) if isinstance(data, dict) else []

    return _normalise(results, max_results)
