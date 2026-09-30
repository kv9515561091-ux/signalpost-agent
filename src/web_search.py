import requests
from bs4 import BeautifulSoup


def fetch_page(url: str) -> dict:
    if not is_safe_url(url):
        raise ValueError("Unsafe URL")

    response = requests.get(
        url,
        timeout=10,
        headers={"User-Agent": "SignalpostAgent/1.0"}
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""

    return {
        "url": url,
        "title": title,
        "content": soup.get_text(" ", strip=True),
    }
from urllib.parse import urlparse


def is_safe_url(url: str) -> bool:
    parsed = urlparse(url)

    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
    )