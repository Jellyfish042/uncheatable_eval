import scrapy
import json
from datetime import datetime, timedelta, timezone
from jsonpath_ng import parse
from crawler.items import BBCNewsItem


class BBCSpider(scrapy.Spider):
    name = "bbc"
    allowed_domains = ["bbc.com"]

    custom_settings = {
        "USER_AGENT": "User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "ITEM_PIPELINES": {
            "crawler.pipelines.SpecialCharFilterPipeline": 50,
            "crawler.pipelines.LengthFilterPipeline": 100,
            "crawler.pipelines.DateRangeFilterPipeline": 150,
            "crawler.pipelines.MinHashLSHDuplicateFilterPipeline": 300,
            "crawler.pipelines.JsonWriterPipeline": 400,
        },
        "CLOSESPIDER_ITEMCOUNT": 500,
        "DOWNLOAD_DELAY": 0.25,
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, start_date="2025-12-01", end_date="2025-12-15", max_samples=1000, *args, **kwargs):
        super(BBCSpider, self).__init__(*args, **kwargs)
        self.start_date = start_date
        self.end_date = end_date
        self.max_samples = int(max_samples)
        self.subtitle = "news"
        self.seen_article_urls = set()
        self.sitemap_indexes = [
            "https://www.bbc.com/sitemaps/https-index-com-news.xml",
            "https://www.bbc.com/sitemaps/https-index-com-archive.xml",
        ]

        self.valid_prefixes = [
            "https://www.bbc.com/news/articles/",
            "https://www.bbc.com/news/world-",
            "https://www.bbc.com/news/uk-",
            "https://www.bbc.com/news/business-",
            "https://www.bbc.com/news/science-",
            "https://www.bbc.com/news/newsbeat-",
            "https://www.bbc.com/news/entertainment-",
            "https://www.bbc.com/news/explainers-",
            "https://www.bbc.com/news/education-",
            "https://www.bbc.com/news/blogs-",
            "https://www.bbc.com/news/health-",
        ]

    async def start(self):
        for url in self.sitemap_indexes:
            self.logger.info(f"Sitemap index: {url}")
            yield scrapy.Request(
                url=url,
                callback=self.parse_sitemap_index,
                dont_filter=True,
            )

    def parse_sitemap_index(self, response):
        sitemap_nodes = response.xpath('//*[local-name()="sitemap"]')

        for sitemap_node in sitemap_nodes:
            loc = sitemap_node.xpath('./*[local-name()="loc"]/text()').get()
            lastmod = sitemap_node.xpath('./*[local-name()="lastmod"]/text()').get()

            if not loc:
                continue

            if lastmod and lastmod[:10] < self.start_date:
                continue

            yield scrapy.Request(url=loc, callback=self.parse_sitemap, dont_filter=True)

    def parse_sitemap(self, response):
        url_nodes = response.xpath('//*[local-name()="url"]')

        for url_node in url_nodes:
            if len(self.seen_article_urls) >= self.max_samples:
                return

            loc = url_node.xpath('./*[local-name()="loc"]/text()').get()
            if not loc:
                continue

            loc = loc.split("?")[0]
            if not any(loc.startswith(prefix) for prefix in self.valid_prefixes):
                continue

            sitemap_date = url_node.xpath('./*[local-name()="news"]/*[local-name()="publication_date"]/text()').get()
            if not sitemap_date:
                sitemap_date = url_node.xpath('./*[local-name()="lastmod"]/text()').get()
            if not sitemap_date:
                continue

            article_date = sitemap_date[:10]
            if article_date < self.start_date or article_date > self.end_date:
                continue

            if loc in self.seen_article_urls:
                continue

            self.seen_article_urls.add(loc)
            yield scrapy.Request(url=loc, callback=self.parse_article, meta={"sitemap_date": sitemap_date})

    def parse_google(self, response):
        all_links = response.css("a::attr(href)").getall()

        bbc_links = []
        for link in all_links:
            if "/url?q=" in link:
                link = link.split("/url?q=")[1].split("&")[0]

            if any(link.startswith(prefix) for prefix in self.valid_prefixes):
                bbc_links.append(link)

        if not bbc_links:
            self.logger.debug(f"No BBC links found on {response.url}")
            return

        for link in set(bbc_links):
            yield scrapy.Request(url=link, callback=self.parse_article, meta={"date": response.meta["date"]})

        current_start = response.meta["start_index"]
        next_start = current_start + 10

        if next_start < 1000:
            next_url = response.url.replace(f"start={current_start}", f"start={next_start}")
            yield scrapy.Request(
                url=next_url, callback=self.parse_google, meta={"date": response.meta["date"], "start_index": next_start}, dont_filter=True
            )

    def parse_article(self, response):
        try:
            script_text = response.css("script#__NEXT_DATA__::text").get()

            if not script_text:
                # self.logger.warning(f"No NEXT_DATA found: {response.url}")
                return

            json_data = json.loads(script_text)

            page_props = json_data.get("props", {}).get("pageProps", {})
            page_data = page_props.get("page", {})

            if not page_data:
                return

            page_key = list(page_data.keys())[0]
            contents = page_data[page_key].get("contents", [])

            iso_date = None
            title = None
            texts = []
            for block in contents:
                if block.get("type") == "headline":
                    jsonpath_expression = parse("$..text")
                    matches = [match.value for match in jsonpath_expression.find(block)]
                    title = matches[0]
                    texts.insert(0, title)
                if block.get("type") == "timestamp":
                    timestamp_ms = block.get("model", {}).get("timestamp")
                    dt_object = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
                    iso_date = dt_object.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                if block.get("type") == "text":
                    text_model = block.get("model", {})
                    for sub_block in text_model.get("blocks", []):
                        sub_model = sub_block.get("model", {})
                        if "text" in sub_model:
                            texts.append(sub_model["text"])
            for _ in range(2):
                if "bbc" in texts[-1].lower() or "sign up" in texts[-1].lower():
                    texts = texts[:-1]
            full_text = "\n".join(texts).strip()
            if full_text:
                item = BBCNewsItem()
                item["content"] = full_text
                item["url"] = response.url
                item["date"] = iso_date or response.meta.get("sitemap_date")
                item["category"] = "bbc_news"
                item["metadata"] = {"title": title}
                yield item

        except Exception as e:
            self.logger.error(f"Error parsing article {response.url}: {e}")

    @staticmethod
    def generate_dates(start_date, end_date):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        step = timedelta(days=1)
        date_list = []
        while start <= end:
            date_list.append(start.strftime("%m/%d/%Y"))
            start += step
        return date_list
