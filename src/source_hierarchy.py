SOURCE_PRIORITY = {
    "official_registry": 100,
    "official_company_website": 90,
    "official_government_source": 85,
    "official_annual_report": 80,
    "reputable_business_database": 60,
    "news_source": 50,
    "search_result": 30,
}


def get_source_priority(source_type: str) -> int:
    return SOURCE_PRIORITY.get(source_type, 0)


def rank_sources(sources: list[dict]) -> list[dict]:
    return sorted(
        sources,
        key=lambda source: get_source_priority(
            source.get("source_type", "")
        ),
        reverse=True,
    )