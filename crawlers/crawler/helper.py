import requests
import logging
import tempfile
import os
import subprocess
import sys
import asyncio


def _get_proxy_server() -> str | None:
    proxy_list = os.environ.get("ROTATING_PROXY_LIST", "")
    for proxy in proxy_list.split(","):
        proxy = proxy.strip()
        if proxy:
            return proxy
    return None


def _download_pdf_via_requests(doi: str) -> tuple[bytes | None, str]:
    url = f"https://www.biorxiv.org/content/{doi}.full.pdf"
    proxy_server = _get_proxy_server()
    proxies = None
    if proxy_server:
        proxies = {
            "http": proxy_server,
            "https": proxy_server,
        }

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            },
            timeout=60,
            allow_redirects=True,
            proxies=proxies,
        )
    except Exception as e:
        proxy_note = f" via proxy {proxy_server}" if proxy_server else ""
        return None, f"direct HTTP request failed{proxy_note}: {e}"

    content_type = response.headers.get("content-type", "")
    if response.status_code != 200:
        return None, f"direct HTTP request returned status {response.status_code} (content-type={content_type})"

    pdf_bytes = response.content
    if not pdf_bytes.startswith(b"%PDF"):
        prefix = pdf_bytes[:200].decode("utf-8", errors="replace").replace("\n", " ")
        return None, f"direct HTTP request returned non-PDF content (content-type={content_type}, prefix={prefix!r})"

    source = "direct HTTP"
    if proxy_server:
        source += f" via proxy {proxy_server}"
    return pdf_bytes, source


def _download_pdf_in_subprocess(doi: str, headless: bool = True) -> tuple[bytes | None, str]:
    """在独立进程中下载 PDF（避免 asyncio 冲突）"""
    proxy_server = _get_proxy_server()
    script = f"""
import tempfile
import os
from playwright.sync_api import sync_playwright

doi = {doi!r}
headless = {headless}
proxy_server = {proxy_server!r}

try:
    pw = sync_playwright().start()
    launch_kwargs = {{"headless": headless}}
    if proxy_server:
        launch_kwargs["proxy"] = {{"server": proxy_server}}
    browser = pw.chromium.launch(**launch_kwargs)
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    page = context.new_page()
    url = f"https://www.biorxiv.org/content/{{doi}}.full.pdf"

    with page.expect_download(timeout=60000) as download_info:
        try:
            page.goto(url)
        except Exception as e:
            if "Download is starting" not in str(e):
                raise

    download = download_info.value

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name

    download.save_as(tmp_path)

    with open(tmp_path, "rb") as f:
        import sys
        sys.stdout.buffer.write(f.read())

    os.unlink(tmp_path)
    page.close()
    browser.close()
    pw.stop()
except Exception as e:
    import sys
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=120)

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        if not stderr:
            stderr = "unknown error"
        return None, f"Playwright fallback failed with exit code {result.returncode}: {stderr}"

    if not result.stdout:
        return None, "Playwright fallback returned empty content"

    return result.stdout, "Playwright fallback"


class PlaywrightPDFDownloader:

    def __init__(self, headless=True):
        self.headless = headless
        self.logger = logging.getLogger("PlaywrightPDFDownloader")

    async def download_pdf(self, doi: str) -> bytes | None:
        loop = asyncio.get_event_loop()

        self.logger.info(f"Downloading PDF for DOI: {doi}")

        try:
            pdf_bytes, source = await loop.run_in_executor(None, _download_pdf_via_requests, doi)
            if pdf_bytes:
                self.logger.info(f"Downloaded PDF for {doi} via {source}, size: {len(pdf_bytes)} bytes")
                return pdf_bytes

            self.logger.warning(f"Direct download failed for {doi}: {source}")

            pdf_bytes, source = await loop.run_in_executor(None, _download_pdf_in_subprocess, doi, self.headless)
            if pdf_bytes:
                self.logger.info(f"Downloaded PDF for {doi} via {source}, size: {len(pdf_bytes)} bytes")
                return pdf_bytes

            self.logger.error(f"Download failed for {doi}: {source}")
            return None
        except Exception as e:
            self.logger.error(f"Download failed for {doi}: {e}")
            return None

    def close(self):
        self.logger.info("Playwright downloader closed")


class MinerUClient:
    def __init__(self, api_url="http://0.0.0.0:8000"):
        self.api_url = api_url.rstrip("/") + "/file_parse"
        self.logger = logging.getLogger("MinerUClient")

    async def process_pdf_stream_async(self, file_name, file_bytes):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_pdf_stream, file_name, file_bytes)

    def process_pdf_stream(self, file_name, file_bytes):
        try:
            files = [("files", (file_name, file_bytes, "application/pdf"))]

            data = {
                "backend": "vlm-vllm-async-engine",
            }

            response = requests.post(self.api_url, files=files, data=data, timeout=600)
            response.raise_for_status()

            result = response.json()

            if "results" in result:
                return result["results"][list(result["results"].keys())[0]].get("md_content", "")
            else:
                self.logger.error(f"Unexpected response structure: {result}")
                return None

        except Exception as e:
            self.logger.error(f"Local API failed for {file_name}: {e}")
            return None
