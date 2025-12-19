import requests
import logging


class MinerUClient:
    def __init__(self, api_url="http://0.0.0.0:8000"):
        self.api_url = api_url.rstrip("/") + "/file_parse"
        self.logger = logging.getLogger("MinerUClient")

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
