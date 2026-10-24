# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

import asyncio
import time

from scrapy import signals
from scrapy.downloadermiddlewares.retry import get_retry_request

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class GithubCrawlerSpiderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, or item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request or item objects.
        pass

    async def process_start(self, start):
        # Called with an async iterator over the spider start() method or the
        # maching method of an earlier spider middleware.
        async for item_or_request in start:
            yield item_or_request

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


class GithubRateLimitMiddleware:
    async def process_response(self, request, response, spider):
        if response.status not in (403, 429):
            return response

        message = response.body[:500].decode("utf-8", errors="replace")
        remaining = response.headers.get("X-RateLimit-Remaining")
        retry_after = response.headers.get("Retry-After")
        if (response.status == 403 and remaining != b"0" and retry_after is None
                and "rate limit" not in message.lower() and "abuse detection" not in message.lower()):
            spider.logger.error("GitHub 403: %s; %s", request.url, message)
            return response

        retry = get_retry_request(request, spider=spider, reason="github_rate_limit", max_retry_times=5)
        if retry is None:
            request.meta["dont_retry"] = True
            return response

        delay = 60 * 2 ** request.meta.get("retry_times", 0)
        try:
            if retry_after is not None:
                delay = max(1, float(retry_after) + 1)
            elif remaining == b"0" and response.headers.get("X-RateLimit-Reset"):
                delay = max(1, float(response.headers["X-RateLimit-Reset"]) - time.time() + 1)
        except ValueError:
            pass

        # With one concurrent request, waiting here also holds back the next request.
        spider.logger.warning("GitHub rate limited; waiting %.0fs before retrying %s", delay, request.url)
        await asyncio.sleep(delay)
        return retry


class GithubCrawlerDownloaderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)
