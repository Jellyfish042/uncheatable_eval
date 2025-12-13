import scrapy


class WikipediaSpider(scrapy.Spider):
    name = "wikipedia"
    allowed_domains = ["temp.com"]
    start_urls = ["https://temp.com"]

    def parse(self, response):
        pass
