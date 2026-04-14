import scrapy
import json
import asyncio
from datetime import datetime
from crawler.items import BiorxivPaperItem
from crawler.helper import MinerUClient, PlaywrightPDFDownloader
import re


class BiorxivSpider(scrapy.Spider):
    name = "biorxiv"
    allowed_domains = ["api.biorxiv.org", "biorxiv.org"]

    custom_settings = {
        "ITEM_PIPELINES": {
            "crawler.pipelines.SpecialCharFilterPipeline": 50,
            "crawler.pipelines.LengthFilterPipeline": 100,
            "crawler.pipelines.DateRangeFilterPipeline": 150,
            "crawler.pipelines.MinHashLSHDuplicateFilterPipeline": 300,
            "crawler.pipelines.JsonWriterPipeline": 400,
        },
        "CLOSESPIDER_ITEMCOUNT": 500,
        "LOG_LEVEL": "INFO",
        "CONCURRENT_REQUESTS": 16,
        "DOWNLOAD_DELAY": 0.5,
        "COOKIES_ENABLED": False,
    }

    def __init__(
        self,
        start_date="2025-12-01",
        end_date="2025-12-15",
        classification="all",
        page_size=100,
        mineru_api="http://0.0.0.0:8000/",
        size_limit=50,
        *args,
        **kwargs,
    ):
        super(BiorxivSpider, self).__init__(*args, **kwargs)

        self.subtitle = "all"
        self.start_date = start_date
        self.end_date = end_date
        self.classification = classification
        self.page_size = int(page_size)
        self.size_limit = size_limit
        self.mineru_client = MinerUClient(api_url=mineru_api)
        self.pdf_downloader = PlaywrightPDFDownloader(headless=True)
        self.item_count = 0
        self.max_items = 500  # 与 CLOSESPIDER_ITEMCOUNT 保持一致

    def start_requests(self):
        """使用 BioRxiv API 获取论文列表"""
        url = f"https://api.biorxiv.org/details/biorxiv/{self.start_date}/{self.end_date}/0"
        yield scrapy.Request(url=url, callback=self.parse_api, meta={"cursor": 0})

    async def parse_api(self, response):
        """解析 API 返回的 JSON 数据"""
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse API response: {e}")
            return

        messages = data.get("messages", [])
        if not messages or messages[0].get("status") != "ok":
            self.logger.error(f"API error: {messages}")
            return

        total = int(messages[0].get("total", 0))
        cursor = response.meta["cursor"]
        self.logger.info(f"API response: cursor={cursor}, total={total}")

        # 1. 收集本页所有符合条件的 paper 元数据
        papers_to_download = []
        collection = data.get("collection", [])
        for paper in collection:
            doi = paper.get("doi")
            category = paper.get("category", "unknown")
            version = paper.get("version", "1")

            # 只爬取首次发表的文章（version == "1"），跳过更新的文章
            if version != "1":
                self.logger.debug(f"Skipping updated paper (version {version}): {paper.get('title', '')}")
                continue

            if self.classification != "all" and category != self.classification:
                continue

            title = paper.get("title", "")
            date_str = paper.get("date", "")

            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                formatted_date = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                formatted_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

            self.logger.info(f"Found paper: {title} | DOI: {doi} | Category: {category}")
            papers_to_download.append(
                {
                    "doi": doi,
                    "title": title,
                    "date": formatted_date,
                    "category": category,
                }
            )

        download_semaphore = asyncio.Semaphore(16)
        ocr_semaphore = asyncio.Semaphore(16)

        async def process_paper(paper):
            """单个论文的完整处理流程：下载 -> OCR -> 返回结果"""
            async with download_semaphore:
                pdf_bytes = await self.pdf_downloader.download_pdf(paper["doi"])

            if not pdf_bytes:
                self.logger.warning(f"Failed to download PDF for {paper['title']} (DOI: {paper['doi']})")
                return None

            file_name = f"{paper['doi'].replace('/', '_')}.pdf"
            self.logger.info(f"PDF downloaded: {file_name}, size: {len(pdf_bytes)} bytes. Doing OCR...")

            async with ocr_semaphore:
                content = await self.mineru_client.process_pdf_stream_async(file_name, pdf_bytes)

            if not content:
                self.logger.warning(f"OCR returned empty content for {file_name}")
                return None

            cleaned_content = self.clean_text(content)
            item = BiorxivPaperItem()
            item["content"] = cleaned_content
            item["category"] = f"biorxiv_{self.subtitle}"
            item["date"] = paper["date"]
            item["url"] = f"https://www.biorxiv.org/content/{paper['doi']}.full.pdf"
            item["metadata"] = {
                "title": paper["title"],
                "raw_content": content,
                "classification": paper["category"],
                "doi": paper["doi"],
            }
            self.logger.info(f"Successfully processed {file_name}")
            return item

        tasks = [process_paper(paper) for paper in papers_to_download]

        for coro in asyncio.as_completed(tasks):
            item = await coro
            if item:
                self.item_count += 1
                yield item
                # 达到目标数量后停止
                if self.item_count >= self.max_items:
                    self.logger.info(f"Reached max items ({self.max_items}), stopping...")
                    return

        # 只有未达到目标数量时才请求下一页
        next_cursor = cursor + self.page_size
        if next_cursor < total and self.item_count < self.max_items:
            next_url = f"https://api.biorxiv.org/details/biorxiv/{self.start_date}/{self.end_date}/{next_cursor}"
            yield scrapy.Request(url=next_url, callback=self.parse_api, meta={"cursor": next_cursor})

    def process_pdf(self, pdf_bytes, file_name, doi, title, date, actual_classification):
        """处理下载的PDF文件"""
        try:
            content = self.mineru_client.process_pdf_stream(file_name, pdf_bytes)

            if content:
                cleaned_content = self.clean_text(content)

                item = BiorxivPaperItem()
                item["content"] = cleaned_content
                item["category"] = f"biorxiv_{self.subtitle}"
                item["date"] = date
                item["url"] = f"https://www.biorxiv.org/content/{doi}.full.pdf"
                item["metadata"] = {
                    "title": title,
                    "raw_content": content,
                    "classification": actual_classification,
                    "doi": doi,
                }
                yield item
                self.logger.info(f"Successfully processed {file_name}")
            else:
                self.logger.warning(f"OCR returned empty content for {file_name}")

        except Exception as e:
            self.logger.error(f"Error processing {file_name}: {e}")

    def closed(self, reason):
        """爬虫关闭时清理资源"""
        self.pdf_downloader.close()
        self.logger.info(f"Spider closed: {reason}")

    @staticmethod
    def clean_text(text):
        """清理文本内容，移除引用、致谢、附录等"""
        # 移除references, citations, acknowledgements
        lines = text.splitlines(keepends=True)
        cut_off_index = len(lines)
        pattern = re.compile(r"^\s*#+\s*(?:[\d\w\.]+\s+)?(?:references?|citations?|acknowledg[e]?ments?)(?:[:\.])?\s*$", re.IGNORECASE)
        for i in range(len(lines) - 1, -1, -1):
            if pattern.match(lines[i].strip()):
                cut_off_index = i
        text = "".join(lines[:cut_off_index])

        # 移除appendix
        target_words = ["appendix", "author", "Bibliography"]
        lines = text.splitlines(keepends=True)
        cut_off_index = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            current_line = lines[i]
            clean_line_lower = current_line.strip().lower()
            if current_line.lstrip().startswith("# ") and any([word.lower() in clean_line_lower for word in target_words]):
                cut_off_index = i
        text = "".join(lines[:cut_off_index])

        # 移除references（编号格式）
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

        # 移除authors（从标题到abstract之间的内容）
        title = text.split("\n")[0]
        target_word = "abstract"
        text_begin = text[: int(len(text) * 0.2)]  # 只检查前20%的文本
        start_index = text_begin.lower().find(target_word)
        if not start_index == -1:
            cut_off_index = start_index
            text = title + "\n\n" + text[cut_off_index:]

        # 移除figures
        text = re.sub(r"(\n\n)?!\[.*?\]\(.*?\)", "", text)

        # 移除cite，支持 [1]、[1, 2]、[1-3]、[1; 2] 等格式
        text = re.sub(r"\s*\[[\d,;\s-]+\]", "", text)

        # 移除©
        text = re.sub(r"(?m)^\s*©.*\n?", "", text)

        return text
