import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse
from helpers import save_json
from proxy import ProxyManager
from datetime import datetime
from lxml import html
import time
import re
from requests.exceptions import RequestException
from queue import Queue
import threading


class AO3Crawler:
    def __init__(self):
        self.queue = Queue()
        self.all_works = set()
        self.lock = threading.Lock()
        self.stop_signal = False

    @staticmethod
    def extract_filtered_work_ids(html):
        soup = BeautifulSoup(html, 'html.parser')
        work_links = soup.select(
            'ol.work.index.group li.work.blurb.group a[href^="/works/"]:not([href*="?"]):not([href*="#"])')
        work_ids = []
        for link in work_links:
            work_url = link.get('href')
            if not work_url.endswith('bookmarks') and not work_url.endswith('collections'):
                if 'chapters' in work_url:
                    work_url = work_url.split('/chapters')[0]
                work_id = work_url.split('/')[-1] if '/' in work_url else ''
                work_ids.append(work_id)
        return work_ids

    @staticmethod
    def calculate_dates(start_date_str, end_date_str):
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        today = datetime.today().date()
        if start_date >= today or end_date >= today:
            return "Error: Dates must be earlier than today."
        days_from_today_to_end = (today - end_date).days
        days_from_today_to_start = (today - start_date).days
        result = f"{days_from_today_to_end}-{days_from_today_to_start}+days"
        return result

    @staticmethod
    def get_work(work_id, retries=3, delay=2, proxies=None):
        url = f'https://archiveofourown.org/works/{work_id}'
        xpaths = [
            '//div[@id="chapters"]/div[@class="userstuff"]//p',
            '//div[@id="chapter-1"]/div[@class="userstuff module"]//p'
        ]

        for attempt in range(retries):
            try:
                response = requests.get(url, proxies=proxies, timeout=(5, 10))
                if response.status_code == 200:
                    tree = html.fromstring(response.content)
                    for xpath in xpaths:
                        paragraphs = tree.xpath(xpath)
                        if paragraphs:
                            result = [paragraph.text_content().replace('\xa0', '\n') for paragraph in paragraphs]
                            result = ''.join(result)
                            result = re.sub(r'\n+', '\n', result)
                            return result.strip()
            except requests.RequestException:
                pass
            time.sleep(delay)
        return []

    @staticmethod
    def fetch_url_with_retries(url, max_retries=3, delay=2, verify=True, proxies=None):
        retries = 0
        while retries < max_retries:
            # print('fetch_url_with_retries', url, retries)
            try:
                response = requests.get(url, verify=verify, proxies=proxies, timeout=(5, 10))
                return response
            except RequestException:
                retries += 1
                time.sleep(delay)
        raise Exception(f"Failed to fetch URL after {max_retries} attempts")

    def producer(self, url_template, set_time, language_id, max_page, manager):
        for page in range(1, max_page + 1):
            if self.stop_signal:
                # print('producer stop')
                break
            url = url_template.replace('<PAGE>', str(page)).replace('<LANGUAGE_ID>', str(language_id)).replace('<TIME>',
                                                                                                               set_time)
            proxy = manager.get_random_proxy()
            response = self.fetch_url_with_retries(url, proxies=proxy)
            if response.status_code == 200:
                content = response.text
                work_ids = self.extract_filtered_work_ids(content)
                # print(work_ids)
                for work_id in work_ids:
                    self.queue.put(work_id)

    def consumer(self, min_char, max_char, manager, pbar, num_works):
        while not self.stop_signal:
            work_id = self.queue.get()
            proxy = manager.get_random_proxy()
            final_text = self.get_work(work_id, proxies=proxy)[:max_char]
            if len(final_text) > min_char:
                with self.lock:
                    self.all_works.add(final_text)
                    pbar.update(1)
                    if len(self.all_works) >= num_works:
                        # print('set stop signal')
                        self.stop_signal = True
            self.queue.task_done()
            if self.stop_signal:
                # print('consumer stop')
                break

    def cleanup_queue(self):
        while not self.queue.empty():
            self.queue.get()
            self.queue.task_done()

    def pipeline(self, start_date, end_date, language_id='1', num_works=1000, max_workers=5, max_page=100,
                 min_char=2000, max_char=5000, manager=None):
        set_time = self.calculate_dates(start_date, end_date)
        url_template = 'https://archiveofourown.org/works/search?commit=Search&page=<PAGE>&work_search[language_id]=<LANGUAGE_ID>&work_search%5Brevised_at=<TIME>&work_search[single_chapter]=0&work_search[sort_column]=created_at&work_search[sort_direction]=desc'
        pbar = tqdm(total=num_works)

        producer_thread = threading.Thread(target=self.producer,
                                           args=(url_template, set_time, language_id, max_page, manager))
        producer_thread.start()

        consumer_threads = []
        for _ in range(max_workers):
            t = threading.Thread(target=self.consumer, args=(min_char, max_char, manager, pbar, num_works))
            t.start()
            consumer_threads.append(t)

        producer_thread.join()
        for t in consumer_threads:
            t.join()

        return list(self.all_works)[:num_works]


LANGUAGE_MAP = {
    'english': '1',
    'chinese': 'zh',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_date', type=str, required=True, help='Start date in the format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date in the format YYYY-MM-DD')
    parser.add_argument('--file_name', type=str, required=True, help='JSON file name')
    parser.add_argument('--language', type=str, default='english', choices=['english', 'chinese'],
                        help='Programming language to filter the repositories.')

    parser.add_argument('--max_works', type=int, default=1000,
                        help='Maximum number of works to crawl. Default is 1000.')
    parser.add_argument('--min_length', type=int, default=2000,
                        help='Minimum length of the files to be considered.')
    parser.add_argument('--max_length', type=int, default=5000,
                        help='Maximum length. Default is 5000 characters.')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Max workers')

    args = parser.parse_args()

    crawler = AO3Crawler()
    proxy_manager = ProxyManager()
    data = crawler.pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        num_works=args.max_works,
        language_id=LANGUAGE_MAP[args.language],
        max_workers=args.max_workers,
        min_char=args.min_length,
        max_char=args.max_length,
        manager=proxy_manager,
    )

    save_json(data, args.file_name)
