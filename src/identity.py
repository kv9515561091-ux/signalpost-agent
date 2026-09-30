from src.web_search import fetch_page


def verify_website(company_name: str, website: str) -> bool:
    if not website.startswith(("http://", "https://")):
        website = "https://" + website

    page = fetch_page(website)

    return company_name.lower() in page["content"].lower()