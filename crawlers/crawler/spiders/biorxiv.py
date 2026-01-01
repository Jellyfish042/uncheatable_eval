import scrapy
import asyncio
from urllib.parse import urlencode
from datetime import datetime
from crawler.items import BiorxivPaperItem
from crawler.helper import MinerUClient
import re


class BiorxivSpider(scrapy.Spider):
    name = "biorxiv"
    allowed_domains = ["biorxiv.org"]

    custom_settings = {
        "ITEM_PIPELINES": {
            "crawler.pipelines.LengthFilterPipeline": 100,
            "crawler.pipelines.MinHashLSHDuplicateFilterPipeline": 300,
            "crawler.pipelines.JsonWriterPipeline": 400,
        },
        "CLOSESPIDER_ITEMCOUNT": 500,
        "LOG_LEVEL": "INFO",
        "CONCURRENT_REQUESTS": 16,
        "ROTATING_PROXY_BAN_POLICY": "crawler.policy.BiorxivBanPolicy",
        "DEFAULT_REQUEST_HEADERS": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
        "DOWNLOAD_DELAY": 3,
        "COOKIES_ENABLED": True,
        # 尝试忽略SSL错误（虽然Scrapy默认不忽略，但可以通过自定义ContextFactory，这里先尝试标准设置）
    }

    def __init__(
        self,
        start_date="2025-12-01",
        end_date="2025-12-15",
        classification="all",
        page_size=200,
        mineru_api="http://0.0.0.0:8000/",
        size_limit=50,
        *args,
        **kwargs,
    ):
        super(BiorxivSpider, self).__init__(*args, **kwargs)

        # BioRxiv使用统一的subtitle
        self.subtitle = "biorxiv"
        self.start_date = start_date
        self.end_date = end_date
        self.classification = classification
        self.page_size = int(page_size)
        self.size_limit = size_limit
        self.mineru_client = MinerUClient(api_url=mineru_api)

    def get_search_url(self, start_idx):
        """
        生成BioRxiv搜索URL
        使用advanced search接口，通过limit_from和limit_to参数指定日期范围
        """
        base_url = "https://www.biorxiv.org/search"

        # 构建搜索查询：使用日期范围过滤
        # BioRxiv的搜索语法：limit_from:YYYY-MM-DD limit_to:YYYY-MM-DD
        search_query = f"limit_from:{self.start_date} limit_to:{self.end_date}"

        params = {
            "query": search_query,
            "sort": "date_posted",  # 按发布日期排序
            "num_results": str(self.page_size),
            "start": str(start_idx),
        }

        # 如果指定了特定分类（非"all"），添加到查询中
        if self.classification != "all":
            # 将分类添加到搜索查询中
            params["query"] = f"{search_query} category:{self.classification}"

        return f"{base_url}?{urlencode(params)}", params

    def start_requests(self):
        url, params = self.get_search_url(0)
        yield scrapy.Request(url=url, callback=self.parse_search, meta={"start_idx": 0, "params": params})

    def parse_search(self, response):
        """解析BioRxiv搜索结果页面"""
        # BioRxiv搜索结果的CSS选择器
        results = response.css("div.highwire-article-citation")

        if not results:
            self.logger.info("No more results found.")
            return

        for paper in results:
            # 提取标题链接（需要访问详情页获取PDF和分类）
            title_link = paper.css("a.highwire-cite-linked-title::attr(href)").get()
            if not title_link:
                continue

            if not title_link.startswith("http"):
                title_link = f"https://www.biorxiv.org{title_link}"

            # 提取标题
            title = paper.css("a.highwire-cite-linked-title::text").get()
            if title:
                title = title.strip()

            # 提取日期 (格式如 "2025.12.30")
            date_text = paper.css("span.highwire-cite-metadata-pages::text").get()
            if date_text:
                # 使用正则提取日期
                date_match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", date_text)
                if date_match:
                    year, month, day = date_match.groups()
                    try:
                        dt = datetime(int(year), int(month), int(day))
                        formatted_date = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except Exception as e:
                        self.logger.warning(f"Failed to parse date from '{date_text}': {e}")
                        formatted_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    formatted_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                formatted_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

            self.logger.info(f"Found paper: {title} | URL: {title_link}")

            # 访问文章详情页以获取PDF链接和分类
            yield scrapy.Request(
                url=title_link,
                callback=self.parse_article_page,
                meta={
                    "title": title,
                    "date": formatted_date,
                },
            )

        # 处理分页
        current_idx = response.meta["start_idx"]
        next_idx = current_idx + self.page_size

        if len(results) > 0:
            url, _ = self.get_search_url(next_idx)
            yield scrapy.Request(url=url, callback=self.parse_search, meta={"start_idx": next_idx, "params": response.meta["params"]})

    def parse_article_page(self, response):
        """解析文章详情页，提取PDF链接和学科分类"""
        title = response.meta["title"]

        # 提取PDF链接
        pdf_link = response.css("a.article-dl-pdf-link::attr(href)").get()
        if not pdf_link:
            # 尝试备用选择器
            pdf_link = response.xpath('//a[contains(@class, "pdf")]/@href').get()

        if pdf_link and not pdf_link.startswith("http"):
            pdf_link = f"https://www.biorxiv.org{pdf_link}"

        # 提取学科分类
        category_tag = response.css(".pane-highwire-article-collection-info .highlight::text").get()
        if not category_tag:
            # 尝试其他可能的选择器
            category_tag = response.css("span.highwire-article-collection-term::text").get()

        if category_tag:
            # 将分类名转换为小写并用下划线替换空格
            actual_classification = category_tag.strip().lower().replace(" ", "_").replace("-", "_")
        else:
            actual_classification = "unknown"

        self.logger.info(f"Article: {title} | Classification: {actual_classification} | PDF: {pdf_link}")

        if pdf_link:
            file_name = pdf_link.split("/")[-1]
            if not file_name.endswith(".pdf"):
                file_name += ".pdf"

            yield scrapy.Request(
                url=pdf_link,
                method="HEAD",
                callback=self.check_pdf_size,
                meta={
                    "file_name": file_name,
                    "source_url": pdf_link,
                    "title": title,
                    "date": response.meta["date"],
                    "actual_classification": actual_classification,
                },
            )
        else:
            self.logger.warning(f"No PDF link found for {title}")

    def check_pdf_size(self, response):
        """检查PDF文件大小"""
        file_name = response.meta["file_name"]
        max_size = self.size_limit * 1024 * 1024  # 转换为字节

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
        """处理下载的PDF文件"""
        file_name = response.meta["file_name"]
        pdf_bytes = response.body

        self.logger.info(f"PDF downloaded: {file_name}, size: {len(pdf_bytes)} bytes. Doing OCR...")

        try:
            content = await asyncio.to_thread(self.mineru_client.process_pdf_stream, file_name, pdf_bytes)

            if content:
                cleaned_content = self.clean_text(content)
                actual_classification = response.meta.get("actual_classification", "unknown")

                item = BiorxivPaperItem()
                item["content"] = cleaned_content
                item["category"] = f"biorxiv_{self.subtitle}"
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
        target_words = ["appendix", "author"]
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

        # 移除cite
        text = re.sub(r"\s*\[[\d,\s-]+\]", "", text)

        # 移除©
        text = re.sub(r"(?m)^\s*©.*\n?", "", text)

        return text
