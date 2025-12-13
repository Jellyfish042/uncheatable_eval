import scrapy
import random
import base64
from datetime import datetime, timedelta
from crawler.items import GithubCodeItem
from urllib.parse import urlencode


class GithubSpider(scrapy.Spider):
    name = "github"
    allowed_domains = ["api.github.com"]

    custom_settings = {
        "ITEM_PIPELINES": {
            "crawler.pipelines.LengthFilterPipeline": 100,
            "crawler.pipelines.GitHubDuplicateFilterPipeline": 200,
            "crawler.pipelines.MinHashLSHDuplicateFilterPipeline": 300,
            "crawler.pipelines.JsonWriterPipeline": 400,
        },
        "MIN_LENGTH": 1000,
        "CUT_OFF_LENGTH": 1e6,
        "CLOSESPIDER_ITEMCOUNT": 500,
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, start_date="2025-11-01", end_date="2025-11-15", language="py", access_token=None, *args, **kwargs):
        super(GithubSpider, self).__init__(*args, **kwargs)

        language_map = {
            "py": "Python",
            "cpp": "C++",
            "js": "JavaScript",
            "ts": "TypeScript",
            "md": None,
        }
        subtitle_map = {
            "py": "python",
            "cpp": "cpp",
            "js": "javascript",
            "ts": "typescript",
            "md": "markdown",
        }
        suffix_map = {
            "py": ".py",
            "cpp": ".cpp",
            "js": ".js",
            "ts": ".ts",
            "md": ".md",
        }

        self.start_date = start_date
        self.end_date = end_date
        self.language = language_map[language]
        self.subtitle = subtitle_map[language]
        self.suffix = suffix_map[language]
        self.access_token = access_token

        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if access_token:
            self.headers["Authorization"] = f"token {access_token}"

    def start_requests(self):
        date_list = self.get_date_list(self.start_date, self.end_date)

        for i in range(len(date_list) - 1):
            real_start = date_list[i]
            real_end = date_list[i]

            if self.language is not None:
                query = f"created:{real_start}..{real_end} language:{self.language}"
            else:
                query = f"created:{real_start}..{real_end}"
            params = {"q": query, "sort": "stars", "order": "desc", "per_page": 100, "page": 1}
            url = f"https://api.github.com/search/repositories?{urlencode(params)}"

            yield scrapy.Request(url=url, headers=self.headers, callback=self.parse_search, meta={"page": 1, "base_url": url})

    def parse_search(self, response):
        data = response.json()
        items = data.get("items", [])

        if not items:
            self.logger.info(f"No more items found for URL: {response.url}")
            return

        for repo in items:
            owner = repo["owner"]["login"]
            repo_name = repo["name"]
            default_branch = repo["default_branch"]
            push_date = repo["pushed_at"]

            tree_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{default_branch}?recursive=1"

            yield scrapy.Request(
                url=tree_url,
                headers=self.headers,
                callback=self.parse_tree,
                meta={"repo_url": repo["html_url"], "repo_name": f"{owner}/{repo_name}", "push_date": push_date, "branch": default_branch},
            )

        current_page = response.meta["page"]
        if current_page < 10 and len(items) == 100:
            next_page = current_page + 1
            next_url = response.meta["base_url"].replace(f"page={current_page}", f"page={next_page}")
            yield scrapy.Request(
                url=next_url, headers=self.headers, callback=self.parse_search, meta={"page": next_page, "base_url": response.meta["base_url"]}
            )

    def parse_tree(self, response):
        data = response.json()
        tree = data.get("tree", [])
        repo_sha = data.get("sha", "")

        target_files = [item for item in tree if item["path"].endswith(self.suffix) and item["type"] == "blob"]

        if target_files:
            selected_file = random.choice(target_files)
            file_path = selected_file["path"]
            repo_name = response.meta["repo_name"]

            content_url = f"https://api.github.com/repos/{repo_name}/contents/{file_path}"
            content_permanent_url = f"https://github.com/{repo_name}/blob/{repo_sha}/{file_path}"

            yield scrapy.Request(
                url=content_url,
                headers=self.headers,
                callback=self.parse_content,
                meta=response.meta | {"content_permanent_url": content_permanent_url},
            )

    def parse_content(self, response):
        data = response.json()
        content_b64 = data.get("content", "")

        if content_b64:
            try:
                content_decoded = base64.b64decode(content_b64).decode("utf-8")
                item = GithubCodeItem()
                item["content"] = content_decoded
                item["category"] = f"github_{self.subtitle}"
                item["date"] = response.meta["push_date"]
                item["url"] = response.meta["content_permanent_url"]
                yield item

            except Exception as e:
                self.logger.error(f"Error decoding content: {e}")

    @staticmethod
    def get_date_list(start_date, end_date):
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        delta = end_date_dt - start_date_dt
        return [(start_date_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]
