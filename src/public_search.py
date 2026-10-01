def build_search_queries(
    company_name: str,
    company_number: str,
) -> list[str]:
    return [
        f'"{company_number}"',
        f'"{company_number}" "{company_name}"',
        f'"{company_name}" kontaktinformasjon',
        f'"{company_name}" adresse telefon',
        f'"{company_name}" official website',
    ]