import scrapy
import asyncio
from urllib.parse import urlencode
from datetime import datetime
from crawler.items import ArxivPaperItem
from crawler.helper import MinerUClient
import re


class ArxivSpider(scrapy.Spider):
    name = "arxiv"
    allowed_domains = ["arxiv.org"]

    custom_settings = {
        "ITEM_PIPELINES": {
            "crawler.pipelines.LengthFilterPipeline": 100,
            "crawler.pipelines.DateRangeFilterPipeline": 150,
            "crawler.pipelines.MinHashLSHDuplicateFilterPipeline": 300,
            "crawler.pipelines.JsonWriterPipeline": 400,
        },
        "CLOSESPIDER_ITEMCOUNT": 500,
        "LOG_LEVEL": "INFO",
        "CONCURRENT_REQUESTS": 32,
        "ROTATING_PROXY_BAN_POLICY": "crawler.policy.ArxivBanPolicy",
    }

    def __init__(
        self,
        start_date="2025-12-01",
        end_date="2025-12-15",
        classification="computer_science",
        page_size=200,
        mineru_api="http://0.0.0.0:8000/",
        size_limit=50,
        *args,
        **kwargs,
    ):
        super(ArxivSpider, self).__init__(*args, **kwargs)

        subtitle_map = {
            "computer_science": "cs",
            "physics": "physics",
            "mathematics": "math",
            "economics": "econ",
            "eess": "eess",
            "q_biology": "qbio",
            "q_finance": "qfin",
            "statistics": "stat",
            "other": "other",
        }

        other_classifications = ["economics", "eess", "q_biology", "q_finance", "statistics"]

        self.start_date = start_date
        self.end_date = end_date
        self.classification = classification
        self.subtitle = subtitle_map[classification]

        if classification == "other":
            self.classifications = other_classifications
        else:
            self.classifications = [classification]
        self.page_size = int(page_size)
        self.size_limit = size_limit
        self.mineru_client = MinerUClient(api_url=mineru_api)

    def get_search_url(self, start_idx):
        base_url = "https://arxiv.org/search/advanced"
        params = {
            "advanced": "1",
            "terms-0-operator": "AND",
            "terms-0-term": "",
            "terms-0-field": "title",
            "classification-include_cross_list": "exclude",
            "date-year": "",
            "date-filter_by": "date_range",
            "date-from_date": self.start_date,
            "date-to_date": self.end_date,
            "date-date_type": "submitted_date_first",
            "abstracts": "hide",
            "size": self.page_size,
            "order": "-announced_date_first",
            "start": str(start_idx),
        }

        for cls in self.classifications:
            params[f"classification-{cls}"] = "y"

        return f"{base_url}?{urlencode(params)}", params

    def start_requests(self):
        url, params = self.get_search_url(0)
        yield scrapy.Request(url=url, callback=self.parse_search, meta={"start_idx": 0, "params": params})

    def parse_search(self, response):
        results = response.css("li.arxiv-result")

        if not results:
            self.logger.info("No more results found.")
            return

        for paper in results:
            pdf_url = paper.xpath('.//a[text()="pdf"]/@href').get()

            title = paper.css("p.title::text").get()
            if title:
                title = title.strip()

            raw_date = paper.css("p.is-size-7").re_first(r"Submitted</span>\s*(.*?);")
            dt = datetime.strptime(raw_date.strip(), "%d %B, %Y")
            date_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Extract primary classification
            primary_class = paper.css(".tags .tag.is-link::text").get()
            if primary_class:
                primary_class = primary_class.strip()
                # Extract the main category (e.g., "eess" from "eess.IV")
                actual_classification = primary_class.split(".")[0]
            else:
                # Fallback to self.classification if extraction fails
                actual_classification = self.classification

            self.logger.info(f"Paper: {title} | Classification: {primary_class} | URL: {pdf_url}")

            if pdf_url:
                file_name = pdf_url.split("/")[-1] + ".pdf"

                yield scrapy.Request(
                    url=pdf_url,
                    method="HEAD",
                    callback=self.check_pdf_size,
                    meta={
                        "file_name": file_name,
                        "source_url": pdf_url,
                        "title": title,
                        "date": date_str,
                        "actual_classification": actual_classification,
                    },
                )

        current_idx = response.meta["start_idx"]
        next_idx = current_idx + self.page_size

        if len(results) > 0:
            url, _ = self.get_search_url(next_idx)
            yield scrapy.Request(url=url, callback=self.parse_search, meta={"start_idx": next_idx, "params": response.meta["params"]})

    def check_pdf_size(self, response):
        file_name = response.meta["file_name"]
        max_size = self.size_limit * 1024 * 1024  # 50MB in bytes

        try:
            content_length = int(response.headers.get("Content-Length", 0))
        except ValueError:
            content_length = 0

        if content_length > max_size:
            size_mb = content_length / (1024 * 1024)
            self.logger.info(f"⚠️ Skipping {file_name}: Size {size_mb:.2f} MB exceeds {self.size_limit}MB limit.")
            return

        if content_length > 0:
            self.logger.info(f"✅ Size check passed for {file_name}: {content_length / 1024 / 1024:.2f} MB. Starting download...")

        yield scrapy.Request(
            url=response.url,
            method="GET",
            callback=self.parse_pdf_downloaded,
            meta=response.meta,
            dont_filter=True,
        )

    async def parse_pdf_downloaded(self, response):
        file_name = response.meta["file_name"]
        pdf_bytes = response.body

        self.logger.info(f"PDF downloaded: {file_name}, size: {len(pdf_bytes)} bytes. Doing OCR...")

        try:
            content = await asyncio.to_thread(self.mineru_client.process_pdf_stream, file_name, pdf_bytes)

            if content:
                cleaned_content = self.clean_text(content)
                # Use actual classification from meta, fallback to self.classification
                actual_classification = response.meta.get("actual_classification", self.classification)

                item = ArxivPaperItem()
                item["content"] = cleaned_content
                item["category"] = f"arxiv_{self.subtitle}"
                item["date"] = response.meta["date"]
                item["url"] = response.meta["source_url"]
                item["metadata"] = {"title": response.meta["title"], "raw_content": content, "classification": actual_classification}
                yield item
                self.logger.info(f"Successfully processed {file_name}")
            else:
                self.logger.warning(f"OCR returned empty content for {file_name}")

        except Exception as e:
            self.logger.error(f"Error processing {file_name}: {e}")

    @staticmethod
    def clean_text(text):
        # remove references, citations, acknowledgements, appendix
        lines = text.splitlines(keepends=True)
        cut_off_index = len(lines)
        pattern = re.compile(r"^\s*#+\s*(?:[\d\w\.]+\s+)?(?:references?|citations?|acknowledg[e]?ments?)(?:[:\.])?\s*$", re.IGNORECASE)
        for i in range(len(lines) - 1, -1, -1):
            if pattern.match(lines[i].strip()):
                cut_off_index = i
        text = "".join(lines[:cut_off_index])

        # remove appendix
        target_words = ["appendix", "author"]
        lines = text.splitlines(keepends=True)
        cut_off_index = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            current_line = lines[i]
            clean_line_lower = current_line.strip().lower()
            if current_line.lstrip().startswith("# ") and any([word.lower() in clean_line_lower for word in target_words]):
                cut_off_index = i
        text = "".join(lines[:cut_off_index])

        # remove references
        lines = text.split("\n")
        cleaned_lines = []
        pattern = re.compile(r"^\[\d+\]")
        for i in range(len(lines)):
            line = lines[i]
            if pattern.match(line.strip()):
                if i + 1 < len(lines) and pattern.match(lines[i + 1].strip()):
                    break
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # remove authors
        title = text.split("\n")[0]
        target_word = "abstract"
        text_begin = text[: int(len(text) * 0.2)]  # only check the first 20% of the text
        start_index = text_begin.lower().find(target_word)
        if not start_index == -1:
            cut_off_index = start_index
            text = title + "\n\n" + text[cut_off_index:]

        # remove figures
        text = re.sub(r"(\n\n)?!\[.*?\]\(.*?\)", "", text)

        # remove html tag
        # text = re.sub(r"<[^>]*>", "", text)

        # remove cite
        text = re.sub(r"\s*\[[\d,\s-]+\]", "", text)

        # remove ©
        text = re.sub(r"(?m)^\s*©.*\n?", "", text)

        return text
