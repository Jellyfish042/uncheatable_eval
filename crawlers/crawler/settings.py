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

# You can add your own proxies to the ROTATING_PROXY_LIST
DOWNLOADER_MIDDLEWARES = {
    "rotating_proxies.middlewares.RotatingProxyMiddleware": 610,
    "rotating_proxies.middlewares.BanDetectionMiddleware": 620,
}
# ROTATING_PROXY_LIST = ["http://127.0.0.1:8890"]
