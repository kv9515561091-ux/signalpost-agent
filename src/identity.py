from src.web_search import fetch_page


def verify_website(company_name: str, website: str) -> dict:
    if not website.startswith(("http://", "https://")):
        website = "https://" + website

    page = fetch_page(website)
    content = page["content"].lower()
    name_match = company_name.lower() in content

    return {
        "verified": name_match,
        "website": website,
        "name_match": name_match,
        "source_url": page["url"],
        "page_title": page["title"],
    }