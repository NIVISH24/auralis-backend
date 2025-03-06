from collections.abc import Generator
from typing import Any, Dict, List, override

from scrapy import Request, Spider
from scrapy.http.response import Response


class WebSpider(Spider):
    name = "webscraper"

    def __init__(self, urls: List[str], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_urls = urls

    @override
    def start_requests(self) -> Generator[Request]:
        """Reads URLs from file and sends requests."""
        for url in self.start_urls:
            yield Request(url=url, callback=self.parse)

    @override
    def parse(self, response: Response) -> Generator[Dict[str, Any]]:
        """Extracts title and text content from the page."""
        yield {
            "url": response.url,
            "title": response.css("title::text").get(),
            "content": " ".join(response.css("p::text").getall()),
        }
