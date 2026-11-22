from typing import Final
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


class WebSearch:
    REGION: Final[str] = "ua-uk"
    SAFESEARCH: Final[str] = "moderate"
    MAX_RESULTS: Final[int] = 3
    REQUEST_TIMEOUT: Final[int] = 10
    MAX_PAGE_CHARS: Final[int] = 3000
    USER_AGENT: Final[str] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    UNWANTED_TAGS: Final[tuple[str, ...]] = (
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "iframe",
    )

    def search(self, query: str) -> list[dict[str, str]]:
        if not query or not query.strip():
            return []

        try:
            with DDGS() as ddgs:
                raw_results = ddgs.text(
                    query,
                    region=self.REGION,
                    safesearch=self.SAFESEARCH,
                    max_results=self.MAX_RESULTS,
                )
                if not raw_results:
                    return []

                results: list[dict[str, str]] = []
                for item in raw_results:
                    results.append(
                        {
                            "title": str(item.get("title", "")).strip(),
                            "url": str(item.get("href", "")).strip(),
                            "snippet": str(item.get("body", "")).strip(),
                        }
                    )
                return results
        except Exception:
            return []

    def _is_valid_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    def read_page(self, url: str) -> str:
        if not self._is_valid_url(url):
            return ""

        try:
            headers = {"User-Agent": self.USER_AGENT}
            response = requests.get(url, headers=headers, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                return ""

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(list(self.UNWANTED_TAGS)):
                tag.decompose()

            lines = [line.strip() for line in soup.get_text(separator="").splitlines()]
            cleaned_text = "".join(line for line in lines if line)

            return cleaned_text[: self.MAX_PAGE_CHARS]
        except Exception:
            return ""


if __name__ == "__main__":
    searcher = WebSearch()
    test_query = "Python остання версія документація"
    print(f"Пошуковий запит: {test_query}")

    found = searcher.search(test_query)
    if not found:
        print("Результатів не знайдено.")
    else:
        for idx, res in enumerate(found, 1):
            print(f"{idx}. {res['title']} -> {res['url']}")

        first_url = found[0]["url"]
        print(f"Завантаження сторінки: {first_url}")
        page_text = searcher.read_page(first_url)
        print("Отриманий текст сторінки:")
        print(page_text)
