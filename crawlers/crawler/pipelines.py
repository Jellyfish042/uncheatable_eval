from scrapy.exceptions import DropItem
import json
import os
from datasketch import MinHash, MinHashLSH


class SpecialCharFilterPipeline:
    """
    Pipeline to filter items containing U+2581 character.
    """

    def process_item(self, item, spider):
        content = item.get("content", "")
        if "\u2581" in content:
            spider.logger.info("Content contains U+2581 character, dropping item.")
            raise DropItem("Content contains U+2581 character.")
        return item


class LengthFilterPipeline:
    def __init__(self, min_len, cut_off_len):
        self.min_len = min_len
        self.cut_off_len = cut_off_len

    @classmethod
    def from_crawler(cls, crawler):
        return cls(min_len=crawler.settings.getint("MIN_LENGTH", 1000), cut_off_len=crawler.settings.getint("CUT_OFF_LENGTH", 1e6))

    def process_item(self, item, spider):
        content = item.get("content", "")

        if len(content) < self.min_len:
            raise DropItem(f"Content length {len(content)} out of range.")

        item["content"] = content[: self.cut_off_len]

        return item


class GitHubDuplicateFilterPipeline:
    def __init__(self):
        self.seen_authors = set()

    def process_item(self, item, spider):
        url = item.get("url", "")
        author = url.replace("https://github.com/", "").split("/")[0]
        if author in self.seen_authors:
            spider.logger.info(f"Author {author} already seen.")
            raise DropItem(f"Author {author} already seen.")
        self.seen_authors.add(author)
        return item


class AO3DuplicateFilterPipeline:
    def __init__(self):
        self.seen_authors = set()

    def process_item(self, item, spider):
        authors = item.get("metadata", []).get("authors", [])
        for author in authors:
            if author in self.seen_authors:
                spider.logger.info(f"Author {author} already seen.")
                raise DropItem(f"Author {author} already seen.")
            else:
                self.seen_authors.add(author)
        return item


class MinHashLSHDuplicateFilterPipeline:
    def __init__(self, threshold=0.9, num_perm=128, ngram_size=5):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.ngram_size = ngram_size

    def _normalize(self, text):
        text = text.lower()
        return text

    def _get_shingles(self, text):
        if len(text) < self.ngram_size:
            return {text}
        shingles = set()
        for i in range(len(text) - self.ngram_size + 1):
            shingles.add(text[i : i + self.ngram_size])
        return shingles

    def compute_minhash(self, text):
        clean_text = self._normalize(text)
        m = MinHash(num_perm=self.num_perm)
        shingles = self._get_shingles(clean_text)
        for s in shingles:
            m.update(s.encode("utf8"))
        return m

    def process_item(self, item, spider):
        content = item.get("content", "")
        m = self.compute_minhash(content)
        result = self.lsh.query(m)
        if result:
            spider.logger.info(f"Duplicate content found: {result[:100]}")
            raise DropItem(f"Duplicate content found.")
        self.lsh.insert(item["url"], m)
        return item


class DateRangeFilterPipeline:
    """
    Pipeline to filter items based on date range.
    Drops items whose date is outside the spider's start_date and end_date.
    """

    def process_item(self, item, spider):
        item_date = item.get("date", "")[:10]  # Extract YYYY-MM-DD
        start_date = getattr(spider, "start_date", None)
        end_date = getattr(spider, "end_date", None)

        if start_date and end_date and item_date:
            if item_date < start_date or item_date > end_date:
                spider.logger.info(f"Date {item_date} out of range [{start_date}, {end_date}].")
                raise DropItem(f"Date {item_date} out of range.")

        return item


class LanguageBalanceCounterPipeline:
    """
    Pipeline to accurately count items per language after all filters.
    Only counts items that have passed all previous filtering pipelines.
    """

    def process_item(self, item, spider):
        # Check if spider has language balancing enabled
        if getattr(spider, 'balance_languages', False):
            lang = item.get('metadata', {}).get('language')
            if lang and hasattr(spider, 'language_item_counts'):
                spider.language_item_counts[lang] += 1

                # Log stats periodically
                total = sum(spider.language_item_counts.values())
                if total % 50 == 0:
                    spider._log_balance_stats()

        return item


class JsonWriterPipeline:
    def __init__(self, target=None):
        self.counter = 0
        self.target = target

    def open_spider(self, spider):
        if not os.path.exists("data"):
            os.makedirs("data")
        main_title = spider.name
        subtitle = spider.subtitle
        date_range = f"{spider.start_date.replace('-', '')}to{spider.end_date.replace('-', '')}"
        file_name = f"{main_title}_{subtitle}_{date_range}.jsonl"
        self.file = open(f"data/{file_name}", "w", encoding="utf-8")

    @classmethod
    def from_crawler(cls, crawler):
        return cls(target=crawler.settings.getint("CLOSESPIDER_ITEMCOUNT", None))

    def process_item(self, item, spider):
        self.counter += 1
        total = getattr(self, "target", None)
        if total:
            percent = (self.counter / total) * 100
            spider.logger.info(f"Progress: {self.counter} / {total} ({percent:.2f}%)")
        else:
            spider.logger.info(f"Progress: {self.counter}")
        line = json.dumps(dict(item), ensure_ascii=False) + "\n"
        self.file.write(line)
        return item

    def close_spider(self, spider):
        self.file.close()
