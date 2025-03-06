import logging
from pathlib import Path

from chromadb import PersistentClient
from itemadapter import ItemAdapter
from scrapy import Item, Spider

from webscraper.items import WebscraperItem


class ChromaDBPipeline:
    def __init__(self) -> None:
        """Initialize ChromaDB client."""

    def process_item(self, item: WebscraperItem, spider: Spider) -> Item:
        """Store scraped data in ChromaDB."""
        adapter = ItemAdapter(item)
        spider.log(f"Storing {adapter['url']} in ChromaDB", log_level=logging.INFO)
        chroma_client = PersistentClient(
            path=(Path.cwd() / "uploads" / spider.name).as_posix()
        )
        collection = chroma_client.get_or_create_collection(
            name="embeddings_db",
        )
        collection.add(
            ids=[item["url"]],  # Use URL as unique ID
            documents=[item["content"]],
            metadatas=[{"title": item["title"]}],
        )
        with open("chromadb.log", "a") as log:
            log.write(f"Stored {adapter['url']}")
        return item
