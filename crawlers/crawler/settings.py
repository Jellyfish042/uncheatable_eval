import os

BOT_NAME = "crawler"
SPIDER_MODULES = ["crawler.spiders"]
NEWSPIDER_MODULE = "crawler.spiders"

ROBOTSTXT_OBEY = False

CONCURRENT_REQUESTS = 16

DOWNLOAD_DELAY = 0.0

MIN_LENGTH = 1000
CUT_OFF_LENGTH = 1e6

ITEM_PIPELINES = {}

FEED_EXPORT_ENCODING = "utf-8"

DOWNLOADER_MIDDLEWARES = {
    "rotating_proxies.middlewares.RotatingProxyMiddleware": 610,
    "rotating_proxies.middlewares.BanDetectionMiddleware": 620,
}
# You can add your own proxies via ROTATING_PROXY_LIST environment variable
_proxy_list_str = os.environ.get("ROTATING_PROXY_LIST", "")
ROTATING_PROXY_LIST = [p.strip() for p in _proxy_list_str.split(",") if p.strip()]
