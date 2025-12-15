import scrapy
from crawler.items import AO3WorkItem
from urllib.parse import urlencode
from datetime import datetime
import re


class AO3Spider(scrapy.Spider):
    name = "ao3"
    allowed_domains = ["archiveofourown.gay", "archiveofourown.org"]

    custom_settings = {
        "ITEM_PIPELINES": {
            "crawler.pipelines.LengthFilterPipeline": 100,
            "crawler.pipelines.AO3DuplicateFilterPipeline": 200,
            "crawler.pipelines.MinHashLSHDuplicateFilterPipeline": 300,
            "crawler.pipelines.JsonWriterPipeline": 400,
        },
        "CLOSESPIDER_ITEMCOUNT": 500,
        "DOWNLOAD_DELAY": 0.1,
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, start_date="2025-11-01", end_date="2025-11-15", language="english", max_page=1000, *args, **kwargs):
        super(AO3Spider, self).__init__(*args, **kwargs)

        language_map = {
            "english": "1",
            "chinese": "zh",
        }

        self.start_date = start_date
        self.end_date = end_date
        self.subtitle = language
        self.language_id = language_map.get(language, "1")
        self.max_page = int(max_page)
        self.base_url = "https://archiveofourown.gay/works/search"
        self.step_size = 20

    def start_requests(self):
        date_query = self.calculate_dates(self.start_date, self.end_date)
        if "Error" in date_query:
            self.logger.error(date_query)
            return

        self.base_params = {
            "commit": "Search",
            "work_search[language_id]": self.language_id,
            "work_search[revised_at]": date_query,
            "work_search[single_chapter]": 0,
            "work_search[sort_column]": "created_at",
            "work_search[sort_direction]": "desc",
        }

        for i in range(1, self.step_size + 1):
            yield self.generate_search_request(page=i)

    def generate_search_request(self, page):
        params = self.base_params.copy()
        params["page"] = page
        url = f"{self.base_url}?{urlencode(params)}"
        # self.logger.info(f"Generating search request for page {page}: {url}")
        return scrapy.Request(url=url, callback=self.parse_search, meta={"page": page})

    def parse_search(self, response):
        current_page = response.meta["page"]

        work_links = response.css('ol.work.index.group li.work.blurb.group a[href^="/works/"]::attr(href)').getall()

        if not work_links:
            self.logger.info(f"No works found on page {current_page}. Stop pagination.")
            return

        for work_url in work_links:
            if "bookmarks" in work_url or "collections" in work_url:
                continue
            if "chapters" in work_url:
                work_url = work_url.split("/chapters")[0]

            full_url = response.urljoin(work_url)

            yield scrapy.Request(url=full_url, callback=self.parse_work)

        next_target_page = current_page + self.step_size
        if next_target_page <= self.max_page:
            yield self.generate_search_request(page=next_target_page)

    def parse_work(self, response):

        xpaths = ['//div[@id="chapters"]/div[@class="userstuff"]//p', '//div[@id="chapter-1"]/div[@class="userstuff module"]//p']

        content_paragraphs = []
        for xpath in xpaths:
            nodes = response.xpath(xpath)
            if nodes:
                content_paragraphs = nodes.xpath("string(.)").getall()
                break

        authors = response.css("h3.byline.heading a::text").getall()
        raw_language = response.css("dd.language::text").get()
        language = raw_language.strip() if raw_language else "Unknown"

        if content_paragraphs:
            text_content = "\n".join([p.replace("\xa0", "") for p in content_paragraphs])
            text_content = re.sub(r"\n+", "\n", text_content).strip()

            item = AO3WorkItem()
            item["content"] = text_content
            item["category"] = f"ao3_{self.subtitle}"
            item["url"] = response.url
            item["date"] = self.extract_date_from_page(response)
            item["metadata"] = {"authors": authors, "language": language}

            yield item

    def extract_date_from_page(self, response):
        date_str = response.css("dd.published::text").get().strip()
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.isoformat()

    @staticmethod
    def calculate_dates(start_date_str, end_date_str):
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            today = datetime.today().date()

            if start_date >= today or end_date >= today:
                return "Error: Dates must be earlier than today."

            days_from_today_to_end = (today - end_date).days
            days_from_today_to_start = (today - start_date).days
            result = f"{days_from_today_to_end}-{days_from_today_to_start} days"
            return result

        except Exception as e:
            return f"Error: {str(e)}"
