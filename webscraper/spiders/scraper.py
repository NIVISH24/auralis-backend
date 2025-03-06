from collections.abc import Generator
from logging import INFO, WARN
from typing import Any, Dict, List, override

from scrapy import Request, Spider
from scrapy.http.response import Response


class WebSpider(Spider):
    def __init__(self, name: str, urls: List[str], *args: Any, **kwargs: Any) -> None:
        super().__init__(name, *args, **kwargs)
        self.start_urls = urls

    @override
    def start_requests(self) -> Generator[Request, None, None]:
        """Reads URLs and sends requests."""
        self.log(f"🚀 Starting to scrape {len(self.start_urls)} URLs", INFO)
        headers = {
            "Host": ["example.com"],
            "Accept-Language": ["en"],
            "Accept-Encoding": ["gzip,deflate"],
            "Accept": [
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            ],
            "User-Agent": [
                "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/44.0.2403.155 Safari/537.36"
            ],
        }

        for url in self.start_urls:
            self.log(f"🌍 Sending request to: {url}", INFO)
            yield Request(
                url=url,
                callback=self.parse,
                errback=self.handle_error,
                dont_filter=True,
                headers=headers,
            )

    def handle_error(self, failure):
        """Logs failed requests."""
        self.log(f"❌ Request failed: {failure}", INFO)
        print(f"❌ Request failed: {failure}")  # Debug output

    @override
    def parse(self, response: Response) -> Generator[Dict[str, Any], None, None]:
        """Extracts title and text content from the page."""
        self.log(f"Scraping {response.url}", INFO)
        item = {
            "url": response.url,
            "title": response.css("title::text").get(),
            "content": " ".join(response.css("p::text").getall()),
        }
        self.log(f"{response.status}", WARN)
        self.log(f"Scraped {item['url']}", INFO)
        yield item
