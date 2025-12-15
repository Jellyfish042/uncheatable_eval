import scrapy


class GithubCodeItem(scrapy.Item):
    content = scrapy.Field()
    category = scrapy.Field()
    url = scrapy.Field()
    date = scrapy.Field()
    metadata = scrapy.Field()


class AO3WorkItem(scrapy.Item):
    content = scrapy.Field()
    category = scrapy.Field()
    url = scrapy.Field()
    date = scrapy.Field()
    metadata = scrapy.Field()


class BBCNewsItem(scrapy.Item):
    content = scrapy.Field()
    category = scrapy.Field()
    url = scrapy.Field()
    date = scrapy.Field()
    metadata = scrapy.Field()


class WikipediaArticleItem(scrapy.Item):
    content = scrapy.Field()
    category = scrapy.Field()
    url = scrapy.Field()
    date = scrapy.Field()
    metadata = scrapy.Field()
