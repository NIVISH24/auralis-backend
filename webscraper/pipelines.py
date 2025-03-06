from scrapy import Item, Spider

from model import create_rag_pipeline


class WebscraperPipeline:
    def process_item(self, item: Item, spider: Spider) -> Item:
        model_name = spider.name
        create_rag_pipeline(model_name)
        return item


class ChromaPipeline:
    def process_item(self, item: Item, spider) -> Item:
        model_name = spider.name
        create_rag_pipeline(model_name)
        return item
