def build_search_queries(
    company_name: str,
    company_number: str,
) -> list[str]:
    """
    Build conservative search queries anchored to the
    Norwegian organization number.
    """

    return [
        f'"{company_number}" "{company_name}"',
        f'"{company_number}"',
        f'"{company_name}" Norway',
        f'"{company_name}" official',
    ]