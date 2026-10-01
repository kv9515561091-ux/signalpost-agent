import os

import requests
from dotenv import load_dotenv

from src.request_tracker import request_tracker

load_dotenv()

TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"


def search_tinyfish(
    query: str,
    max_results: int = 10,
) -> list[dict]:
    api_key = os.getenv("TINYFISH_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TINYFISH_API_KEY is not configured"
        )

    request_tracker.record()

    response = requests.get(
        TINYFISH_SEARCH_URL,
        params={
            "query": query,
            "location": "NO",
            "language": "no",
        },
        headers={
            "X-API-Key": api_key,
        },
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])

    normalized = []

    for result in results[:max_results]:
        url = result.get("url")

        if not url:
            continue

        normalized.append(
            {
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("snippet", ""),
            }
        )

    return normalized