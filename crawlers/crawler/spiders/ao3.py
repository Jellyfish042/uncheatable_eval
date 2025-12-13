import scrapy


class Ao3Spider(scrapy.Spider):
    name = "ao3"
    allowed_domains = ["temp.com"]
    start_urls = ["https://temp.com"]

    def parse(self, response):
        pass
