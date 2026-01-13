import scrapy
import re
from collections import defaultdict
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
            "crawler.pipelines.LanguageBalanceCounterPipeline": 350,
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
        "russian": {"url": "https://ru.wikipedia.org/w/api.php", "variant": None},
        "italian": {"url": "https://it.wikipedia.org/w/api.php", "variant": None},
        "portuguese": {"url": "https://pt.wikipedia.org/w/api.php", "variant": None},
        "korean": {"url": "https://ko.wikipedia.org/w/api.php", "variant": None},
        "turkish": {"url": "https://tr.wikipedia.org/w/api.php", "variant": None},
        "polish": {"url": "https://pl.wikipedia.org/w/api.php", "variant": None},
        "dutch": {"url": "https://nl.wikipedia.org/w/api.php", "variant": None},
        "swedish": {"url": "https://sv.wikipedia.org/w/api.php", "variant": None},
        "indonesian": {"url": "https://id.wikipedia.org/w/api.php", "variant": None},
        "hindi": {"url": "https://hi.wikipedia.org/w/api.php", "variant": None},
        "persian": {"url": "https://fa.wikipedia.org/w/api.php", "variant": None},
        "vietnamese": {"url": "https://vi.wikipedia.org/w/api.php", "variant": None},
        "thai": {"url": "https://th.wikipedia.org/w/api.php", "variant": None},
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
        "\nVéase también",
        "\n↑ ",
        "\nWeblinks",
        "\nEinzelnachweise",
        "\nLiteratur",
        "\nQuellen und Literatur",
        "\nRéférences",
        "\nLiens externes",
        "\nArticles connexes",
        "\nNotes et références",
        "\nSee also",
        "\nNotes",
        "\nVoir aussi",
        "\nBibliographie",
        "\nSiehe auch",
        "Références",
        "المراجع",
        "المصادر",
        "المصادn",
        "انظر أيضًا",
        "مراجع",
        "مصادر",
        "\n参考文献",
        "\n関連項目",
        "\n外部リンク",
        "\n出典",
        "\n脚注",
        "\nСм. также",
        "\nЛитература",
        "\nПримечания",
        "\nСсылки",
        "\nBibliografia",
        "\nVedi anche",
        "\nNote",
        "\nRiferimenti",
        "\nLigações externas",
        "\nVer também",
        "\nNotas",
        "\n참고문헌",
        "\n같이 보기",
        "\n외부 링크",
        "\n주석",
        "\nKaynakça",
        "\nAyrıca bakınız",
        "\nDış bağlantılar",
        "\nBibliografia",
        "\nZobacz też",
        "\nPrzypisy",
        "\nLinki zewnętrzne",
        "\nLiteratuur",
        "\nZie ook",
        "\nNoten",
        "\nExterne links",
        "\nReferensi",
        "\nLihat juga",
        "\nPranala luar",
        "\nसन्दर्भ",
        "\nयह भी देखें",
        "\nबाहरी कड़ियाँ",
        "\nپیوند به بیرون",
        "\nمنابع",
        "\nجستارهای وابسته",
        "\nXem thêm",
        "\nTham khảo",
        "\nLiên kết ngoài",
        "\nอ้างอิง",
        "\nดูเพิ่ม",
        "\nแหล่งข้อมูลอื่น",
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
        self.subtitle = language

        # Language balancing: track item counts per language for priority-based balancing
        self.language_item_counts = defaultdict(int)
        self.balance_languages = language in ("nonenglish", "all")
        if self.balance_languages:
            # Date range queue for each language (for efficient request scheduling)
            self.language_date_queues = {}
            date_list = self.get_date_list(start_date, end_date)
            date_ranges = [(date_list[i], date_list[i + 1]) for i in range(len(date_list) - 1)]
            for lang in self.languages:
                self.language_date_queues[lang] = list(date_ranges)

            # Exhaustion detection: track pending requests and empty responses
            self.language_exhausted = set()
            self.language_pending_requests = defaultdict(int)
            self.language_empty_responses = defaultdict(int)

            # Quota management: dynamic quota per language
            self.total_target = self.custom_settings.get("CLOSESPIDER_ITEMCOUNT", 500)
            self.base_quota_per_language = self.total_target // len(self.languages)
            self.language_quota = {lang: self.base_quota_per_language for lang in self.languages}

            # Configuration
            self.EMPTY_RESPONSE_THRESHOLD = 3

            self.logger.info(f"Language balancing enabled: {len(self.languages)} languages, "
                           f"{self.base_quota_per_language} items per language")

    def start_requests(self):
        if self.balance_languages:
            # Balanced mode: send only one initial request per language
            for lang in self.languages:
                for request in self._create_date_range_request(lang):
                    yield request
        else:
            # Normal mode: send all requests at once
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

    def _create_date_range_request(self, lang):
        """Create a request for the next date range of a language. Returns a list of requests."""
        # Check if language is exhausted
        if lang in self.language_exhausted:
            return []

        # Check if language has reached its quota (considering pending requests)
        effective_count = self.language_item_counts[lang] + self.language_pending_requests[lang]
        if effective_count >= self.language_quota.get(lang, float('inf')):
            return []

        # Check if date queue is empty
        if not self.language_date_queues.get(lang):
            self._check_language_exhaustion(lang)
            return []

        start_date, end_date = self.language_date_queues[lang].pop(0)
        api_url = self.LANGUAGE_CONFIG[lang]["url"]
        variant = self.LANGUAGE_CONFIG[lang]["variant"]

        rcstart = f"{start_date}T00:00:00Z"
        rcend = f"{end_date}T00:00:00Z"

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
        priority = self._calculate_priority(lang)

        # Track pending request
        self.language_pending_requests[lang] += 1

        return [scrapy.Request(
            url=url,
            callback=self.parse_recent_changes,
            meta={"base_params": params, "lang": lang, "api_url": api_url, "variant": variant},
            priority=priority,
        )]

    def _calculate_priority(self, lang):
        """Calculate request priority based on quota completion ratio."""
        if lang in self.language_exhausted:
            return -10000  # Lowest priority for exhausted languages

        current = self.language_item_counts[lang]
        quota = self.language_quota.get(lang, self.base_quota_per_language)
        completion_ratio = current / quota if quota > 0 else 1.0

        # Higher priority (larger number) for languages with lower completion ratio
        return int((1 - completion_ratio) * 1000)

    def _mark_language_exhausted(self, lang, reason=""):
        """Mark a language as exhausted and trigger quota redistribution."""
        if lang not in self.language_exhausted:
            self.language_exhausted.add(lang)
            self.logger.info(f"Language '{lang}' marked as exhausted. Reason: {reason}. "
                           f"Items collected: {self.language_item_counts[lang]}")
            self._redistribute_quota()

    def _check_language_exhaustion(self, lang):
        """Check if a language should be marked as exhausted."""
        if lang in self.language_exhausted:
            return True

        # Condition 1: Date queue is empty
        queue_empty = not self.language_date_queues.get(lang)

        # Condition 2: No pending requests
        no_pending = self.language_pending_requests[lang] <= 0

        # Condition 3: Too many consecutive empty responses
        too_many_empty = self.language_empty_responses[lang] >= self.EMPTY_RESPONSE_THRESHOLD

        if queue_empty and no_pending:
            self._mark_language_exhausted(lang, "date queue empty and no pending requests")
            return True

        if too_many_empty:
            self._mark_language_exhausted(lang, f"consecutive empty responses >= {self.EMPTY_RESPONSE_THRESHOLD}")
            return True

        return False

    def _redistribute_quota(self):
        """Redistribute quota from exhausted languages to active ones."""
        active_languages = [l for l in self.languages if l not in self.language_exhausted]

        if not active_languages:
            self.logger.warning("All languages exhausted!")
            return

        # Calculate remaining quota to distribute
        current_total = sum(self.language_item_counts.values())
        remaining = max(0, self.total_target - current_total)

        # Distribute remaining quota, ensuring at least 1 per language if there's any remaining
        base_extra = remaining // len(active_languages) if active_languages else 0
        remainder = remaining % len(active_languages) if active_languages else 0

        for i, lang in enumerate(active_languages):
            # Give first 'remainder' languages an extra 1
            extra = base_extra + (1 if i < remainder else 0)
            self.language_quota[lang] = self.language_item_counts[lang] + extra

        self.logger.info(f"Quota redistributed. Active: {len(active_languages)}, "
                        f"Exhausted: {len(self.language_exhausted)}, Remaining: {remaining}")

        # Proactively create requests for idle languages that got extra quota
        if hasattr(self, 'crawler') and self.crawler:
            for lang in active_languages:
                if self.language_pending_requests[lang] == 0:
                    if self.language_item_counts[lang] < self.language_quota.get(lang, 0):
                        for request in self._create_date_range_request(lang):
                            self.crawler.engine.crawl(request)

    def _log_balance_stats(self):
        """Log language balance statistics."""
        stats = []
        for lang in sorted(self.languages):
            count = self.language_item_counts[lang]
            quota = self.language_quota.get(lang, 0)
            status = "EXHAUSTED" if lang in self.language_exhausted else "active"
            stats.append(f"{lang}: {count}/{quota} ({status})")

        self.logger.info(f"Language balance stats: {', '.join(stats)}")

    def parse_recent_changes(self, response):
        data = response.json()
        lang = response.meta["lang"]
        api_url = response.meta["api_url"]
        variant = response.meta["variant"]

        # Decrease pending count for this response
        if self.balance_languages:
            self.language_pending_requests[lang] = max(0, self.language_pending_requests[lang] - 1)

        if "error" in data:
            self.logger.error(f"API Error for {lang}: {data['error']}")
            if self.balance_languages:
                self._check_language_exhaustion(lang)
            return

        website_prefix = api_url.replace("/w/api.php", "")
        recent_changes = data.get("query", {}).get("recentchanges", [])

        # Track empty responses for exhaustion detection
        if self.balance_languages:
            if not recent_changes:
                self.language_empty_responses[lang] += 1
            else:
                self.language_empty_responses[lang] = 0

        for change in recent_changes:
            # Check if language has reached its quota (considering pending requests)
            if self.balance_languages:
                effective_count = self.language_item_counts[lang] + self.language_pending_requests[lang]
                if effective_count >= self.language_quota.get(lang, float('inf')):
                    break

            title = change["title"]
            timestamp = change["timestamp"]

            content_params = {"action": "parse", "page": title, "prop": "text", "format": "json"}
            if variant:
                content_params["variant"] = variant

            content_url = f"{api_url}?{urlencode(content_params)}"

            priority = self._calculate_priority(lang) if self.balance_languages else 0

            # Track pending content request
            if self.balance_languages:
                self.language_pending_requests[lang] += 1

            yield scrapy.Request(
                url=content_url,
                callback=self.parse_content,
                meta={"title": title, "date": timestamp, "url": f"{website_prefix}/wiki/{title.replace(' ', '_')}", "lang": lang},
                priority=priority,
            )

        if "continue" in data:
            continue_params = data["continue"]
            next_params = response.meta["base_params"].copy()
            next_params.update(continue_params)

            next_url = f"{api_url}?{urlencode(next_params)}"
            priority = self._calculate_priority(lang) if self.balance_languages else 0

            # Track pending continue request
            if self.balance_languages:
                self.language_pending_requests[lang] += 1

            yield scrapy.Request(
                url=next_url,
                callback=self.parse_recent_changes,
                meta={"base_params": next_params, "lang": lang, "api_url": api_url, "variant": variant},
                priority=priority,
            )
        elif self.balance_languages:
            # No more pages for this date range, schedule next date range for this language
            for next_request in self._create_date_range_request(lang):
                yield next_request
            # Check if language should be marked as exhausted
            if not self.language_date_queues.get(lang):
                self._check_language_exhaustion(lang)

    def parse_content(self, response):
        data = response.json()
        lang = response.meta["lang"]

        # Decrease pending count for this response
        if self.balance_languages:
            self.language_pending_requests[lang] = max(0, self.language_pending_requests[lang] - 1)

            # Replenish requests: only trigger when pending reaches 0 to avoid over-consuming date queue
            pending = self.language_pending_requests[lang]
            if pending == 0 and lang not in self.language_exhausted:
                effective_count = self.language_item_counts[lang]
                quota = self.language_quota.get(lang, float('inf'))
                if effective_count < quota:
                    for next_request in self._create_date_range_request(lang):
                        yield next_request

            # Check if language should be marked as exhausted
            self._check_language_exhaustion(lang)

        if "error" in data or "parse" not in data:
            return

        html_content = data["parse"]["text"]["*"]

        soup = BeautifulSoup(html_content, "html.parser")

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
        item["category"] = f"wikipedia_{self.language}"
        item["date"] = response.meta["date"]
        item["url"] = response.meta["url"]
        item["metadata"] = {
            "title": response.meta["title"],
            "language": response.meta["lang"],
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
