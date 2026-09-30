from typing import Protocol


class SearchProvider(Protocol):
    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        ...


def normalize_results(results: list[dict]) -> list[dict]:
    return [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet", ""),
        }
        for result in results
        if result.get("url")
    ]