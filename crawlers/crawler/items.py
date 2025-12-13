# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class GithubCodeItem(scrapy.Item):
    content = scrapy.Field()
    category = scrapy.Field()
    url = scrapy.Field()
    date = scrapy.Field()
    metadata = scrapy.Field()
