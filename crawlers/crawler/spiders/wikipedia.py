import scrapy
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from crawler.items import WikipediaArticleItem


class WikipediaSpider(scrapy.Spider):
    name = "wikipedia"

    custom_settings = {
        "ITEM_PIPELINES": {
            "crawler.pipelines.LengthFilterPipeline": 100,
            "crawler.pipelines.MinHashLSHDuplicateFilterPipeline": 300,
            "crawler.pipelines.JsonWriterPipeline": 400,
        },
        "CLOSESPIDER_ITEMCOUNT": 500,
        "LOG_LEVEL": "INFO",
        "CONCURRENT_REQUESTS": 16,
        "DOWNLOAD_DELAY": 0.1,
    }

    LANGUAGE_CONFIG = {
        "english": {"url": "https://en.wikipedia.org/w/api.php", "variant": None},
        "chinese": {"url": "https://zh.wikipedia.org/w/api.php", "variant": "zh-cn"},
        "japanese": {"url": "https://ja.wikipedia.org/w/api.php", "variant": None},
        "spanish": {"url": "https://es.wikipedia.org/w/api.php", "variant": None},
        "german": {"url": "https://de.wikipedia.org/w/api.php", "variant": None},
        "french": {"url": "https://fr.wikipedia.org/w/api.php", "variant": None},
        "arabic": {"url": "https://ar.wikipedia.org/w/api.php", "variant": None},
    }

    STOP_WORDS = [
        "\n参考",
        "\n注释",
        "\n注脚",
        "\n脚注",
        "\n参考资料",
        "\n参考文献",
        "\n参考来源",
        "\n资料来源",
        "\n参见",
        "\n外部链接",
        "\nReferences",
        "\n来源",
        "\n^ ",
        "\nReferencias",
        "\n↑ ",
        "\nWeblinks",
        "\nEinzelnachweise",
        "\nLiteratur",
        "\nRéférences",
        "\nLiens externes",
        "\nArticles connexes",
        "\nNotes et références",
        "\nSee also",
        "\nNotes",
        "Références",
        "المراجع",
        "المصادر",
        "المصادn",
        "انظر أيضًا",
        "مراجع",
        "مصادر",
    ]

    def __init__(self, start_date="2025-12-01", end_date="2025-12-14", language="english", *args, **kwargs):
        super(WikipediaSpider, self).__init__(*args, **kwargs)

        if language == "nonenglish":
            self.languages = [lang for lang in self.LANGUAGE_CONFIG.keys() if lang != "english"]
        elif language == "all":
            self.languages = list(self.LANGUAGE_CONFIG.keys())
        else:
            if language not in self.LANGUAGE_CONFIG:
                raise ValueError(f"Language '{language}' is not supported.")
            self.languages = [language]

        self.start_date = start_date
        self.end_date = end_date
        self.language = language

    def start_requests(self):
        date_list = self.get_date_list(self.start_date, self.end_date)

        for lang in self.languages:
            api_url = self.LANGUAGE_CONFIG[lang]["url"]
            variant = self.LANGUAGE_CONFIG[lang]["variant"]

            for i in range(len(date_list) - 1):
                rcstart = f"{date_list[i]}T00:00:00Z"
                rcend = f"{date_list[i+1]}T00:00:00Z"

                params = {
                    "action": "query",
                    "list": "recentchanges",
                    "rcstart": rcstart,
                    "rcend": rcend,
                    "rcdir": "newer",
                    "rctype": "new",
                    "rcprop": "title|timestamp",
                    "rcnamespace": "0",
                    "rclimit": 500,
                    "format": "json",
                }

                url = f"{api_url}?{urlencode(params)}"
                yield scrapy.Request(url=url, callback=self.parse_recent_changes, meta={"base_params": params, "lang": lang, "api_url": api_url, "variant": variant})

    def parse_recent_changes(self, response):
        data = response.json()

        if "error" in data:
            self.logger.error(f"API Error: {data['error']}")
            return

        lang = response.meta["lang"]
        api_url = response.meta["api_url"]
        variant = response.meta["variant"]
        website_prefix = api_url.replace("/w/api.php", "")

        recent_changes = data.get("query", {}).get("recentchanges", [])
        for change in recent_changes:
            title = change["title"]
            timestamp = change["timestamp"]

            content_params = {"action": "parse", "page": title, "prop": "text", "format": "json"}
            if variant:
                content_params["variant"] = variant

            content_url = f"{api_url}?{urlencode(content_params)}"

            yield scrapy.Request(
                url=content_url,
                callback=self.parse_content,
                meta={"title": title, "date": timestamp, "url": f"{website_prefix}/wiki/{title.replace(' ', '_')}", "lang": lang},
            )

        if "continue" in data:
            continue_params = data["continue"]
            next_params = response.meta["base_params"].copy()
            next_params.update(continue_params)

            next_url = f"{api_url}?{urlencode(next_params)}"
            yield scrapy.Request(url=next_url, callback=self.parse_recent_changes, meta={"base_params": next_params, "lang": lang, "api_url": api_url, "variant": variant})

    def parse_content(self, response):
        data = response.json()

        if "error" in data or "parse" not in data:
            return

        html_content = data["parse"]["text"]["*"]

        soup = BeautifulSoup(html_content, "html.parser")

        # self.logger.info("-" * 100)
        # self.logger.info(f"Soup: {soup.prettify()}")
        # self.logger.info("-" * 100)

        if soup.find(class_="redirectMsg"):
            return

        references = soup.find("span", {"id": "References"})
        if references:
            if references.parent:
                for sibling in references.parent.find_next_siblings():
                    sibling.decompose()
                references.parent.decompose()

        for tag in soup.find_all(["table", "math"]):
            tag.decompose()

        text = soup.get_text()

        for word in self.STOP_WORDS:
            index = text.find(word)
            if index != -1:
                text = text[:index]

        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\[.*?\]", "", text)
        text = response.meta["title"] + "\n" + text.strip()

        item = WikipediaArticleItem()
        item["content"] = text
        item["category"] = f"wikipedia_{response.meta['lang']}"
        item["date"] = response.meta["date"]
        item["url"] = response.meta["url"]
        item["metadata"] = {
            "title": response.meta["title"],
            "entry_created_at": response.meta["date"],
            "crawled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        yield item

    @staticmethod
    def get_date_list(start_date, end_date):
        try:
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
            delta = end_date_dt - start_date_dt
            return [(start_date_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]
        except ValueError as e:
            raise ValueError(f"Date format error. Use YYYY-MM-DD. Error: {e}")
