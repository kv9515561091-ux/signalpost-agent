import os

import requests
from dotenv import load_dotenv


load_dotenv()


def normalize_results(results: list[dict]) -> list[dict]:
    """Normalize search results into a consistent structure."""
    return [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet", ""),
        }
        for result in results
        if result.get("url")
    ]


def filter_company_results(
    results: list[dict],
    company_name: str,
    company_number: str,
) -> list[dict]:
    """
    Keep only results that contain the verified company name
    or Norwegian organization number.
    """

    company_name = company_name.lower().strip()
    company_number = company_number.lower().strip()

    filtered = []

    for result in results:
        text = " ".join(
            [
                result.get("title", ""),
                result.get("url", ""),
                result.get("snippet", ""),
            ]
        ).lower()

        if company_number in text or company_name in text:
            filtered.append(result)

    return filtered


def search_web(
    query: str,
    max_results: int = 5,
    company_name: str | None = None,
    company_number: str | None = None,
) -> list[dict]:
    """
    Search the web using LangSearch.

    If company_name and company_number are provided,
    irrelevant results are filtered out.
    """

    api_key = os.getenv("LANGSEARCH_API_KEY")

    if not api_key:
        raise RuntimeError(
            "LANGSEARCH_API_KEY is not configured"
        )

    response = requests.post(
        "https://api.langsearch.com/v1/web-search",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": max_results,
        },
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    results = (
        data.get("data", {})
        .get("webPages", {})
        .get("value", [])
    )

    normalized = []

    for item in results:
        normalized.append(
            {
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            }
        )

    normalized_results = normalize_results(normalized)

    if company_name and company_number:
        return filter_company_results(
            normalized_results,
            company_name,
            company_number,
        )

    return normalized_results