import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime, timedelta
import re
import argparse
from helpers import save_json
from proxy import ProxyManager
import json
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed


class BBCCrawler:
    def __init__(self):
        self.stop_event = threading.Event()
        pass

    @staticmethod
    def generate_dates_m_d_y_compatible(start_date, end_date):
        start = datetime.strptime(start_date, "%m/%d/%Y")
        end = datetime.strptime(end_date, "%m/%d/%Y")
        step = timedelta(days=1)

        date_list = []
        while start <= end:
            date_str = f"{start.month}/{start.day}/{start.year}"
            date_list.append(date_str)
            start += step

        return date_list

    @staticmethod
    def start_with_any(target_string, string_list):
        for substring in string_list:
            if target_string.startswith(substring):
                return True
        return False

    def get_search_results(self, url, retries=5, sleep_time=1, proxy=None):

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        }

        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, proxies=proxy, timeout=5, verify=True)

                if response.status_code == 200:

                    string_list = ['https://www.bbc.com/news/articles/',
                                   'https://www.bbc.com/news/world-',
                                   'https://www.bbc.com/news/uk-',
                                   'https://www.bbc.com/news/business-',
                                   'https://www.bbc.com/news/science-',
                                   'https://www.bbc.com/news/newsbeat-',
                                   'https://www.bbc.com/news/entertainment-',
                                   'https://www.bbc.com/news/explainers-',
                                   'https://www.bbc.com/news/education-',
                                   'https://www.bbc.com/news/blogs-',
                                   'https://www.bbc.com/news/health-',
                                   ]

                    soup = BeautifulSoup(response.text, 'html.parser')
                    search_results = soup.find_all('a')

                    links = [link['href'] for link in search_results if link.has_attr('href')]
                    links = [x for x in links if self.start_with_any(x, string_list)]
                    # links = [x for x in links if '/live/' not in x]
                    # links = [x for x in links if '/av/' not in x]
                    # links = [x.replace('/url?q=', '') for x in links]
                    # links = [x.split('&')[0] for x in links]
                    # links = [x for x in links if x[-1].isdigit()]

                    return links
                else:
                    # print(f'Response status code {response.status_code}. Retrying in {sleep_time} seconds...')
                    time.sleep(sleep_time)
            except Exception as e:
                # print(f"Error: {e}")
                # print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

        print("Failed to retrieve data after multiple attempts.")
        return []

    @staticmethod
    def convert_date_format(date_str):
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        formatted_date = date_obj.strftime('%m/%d/%Y').replace('/0', '/').lstrip('0')

        return formatted_date

    @staticmethod
    def extract_content_from_url(url, retries=3, delay=2):
        for attempt in range(retries):
            try:
                response = requests.get(url)
                response.raise_for_status()  # Check if the request was successful
                html_content = response.content

                soup = BeautifulSoup(html_content, 'html.parser')

                script_tag = soup.find('script', {'id': '__NEXT_DATA__', 'type': 'application/json'})
                if script_tag:
                    json_data = json.loads(script_tag.string)
                else:
                    return None

                texts = []
                for x in \
                        json_data['props']['pageProps']['page'][
                            list(json_data['props']['pageProps']['page'].keys())[0]][
                            'contents']:
                    if x['type'] in ['text']:
                        for y in x['model']['blocks']:
                            texts.append(y['model']['text'])
                return '\n'.join(texts)
            except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
                # print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    return None

    def fetch_url(self, url, min_length=2000, max_length=5000):
        extracted_contents = self.extract_content_from_url(url)
        if extracted_contents is None:
            return None
        if len(extracted_contents) > min_length:
            return extracted_contents[:max_length]
        else:
            return None

    def producer(self, dates, search_url, max_page_num, no_content_page_allowed, url_queue, proxy_manager):
        for date in dates:
            for url in search_url:
                no_content_page = 0
                for start in [str(x * 10) for x in range(max_page_num)]:
                    if self.stop_event.is_set():
                        return
                    request_url = url.replace('<START>', start).replace('<DATE>', date)
                    proxy = proxy_manager.get_random_proxy()
                    search_links = self.get_search_results(request_url, proxy=proxy)
                    if len(search_links) == 0:
                        no_content_page += 1
                    if no_content_page == no_content_page_allowed:
                        break
                    for link in search_links:
                        url_queue.put(link)

    def consumer(self, url_queue, min_length, max_length, all_news, pbar, max_samples):
        while not self.stop_event.is_set():
            try:
                url = url_queue.get()
                if url is None:
                    break
                content = self.fetch_url(url, min_length, max_length)
                if content and not self.stop_event.is_set():
                    all_news.append(content)
                    pbar.update(1)
                    # print('-' * 100)
                    # print(f'total: {len(all_news)}')
                    if len(all_news) >= max_samples:
                        self.stop_event.set()
                        break
            except Exception as exc:
                print(f'{url} generated an exception: {exc}')
            finally:
                url_queue.task_done()

    def pipeline(self, start_date, end_date, max_samples=1000, min_length=2000, max_length=5000, max_workers=16, proxy_manager=None):
        start_date = self.convert_date_format(start_date)
        end_date = self.convert_date_format(end_date)
        search_url = [
            'https://www.google.com/search?q=news+site:bbc.com/news&tbs=cdr:1,cd_min:<DATE>,cd_max:<DATE>&tbm=nws&start=<START>',
        ]
        max_page_num = 100
        no_content_page_allowed = 5
        all_news = []
        dates = self.generate_dates_m_d_y_compatible(start_date, end_date)
        url_queue = Queue()
        producer_thread = threading.Thread(target=self.producer, args=(dates, search_url, max_page_num, no_content_page_allowed, url_queue, proxy_manager))
        producer_thread.start()

        pbar = tqdm(total=max_samples)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            consumers = [executor.submit(self.consumer, url_queue, min_length, max_length, all_news, pbar, max_samples) for _ in range(max_workers)]
            producer_thread.join()
            # url_queue.join()

            for _ in range(max_workers):
                url_queue.put(None)

            for future in as_completed(consumers):
                future.result()

        pbar.close()
        return all_news[:max_samples]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_date', type=str, required=True,
                        help='The start date (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, required=True,
                        help='The end date (YYYY-MM-DD).')
    parser.add_argument('--file_name', type=str, required=True,
                        help='JSON file name')

    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Number of news to crawl. Default is 1000.')
    parser.add_argument('--min_length', type=int, default=2000,
                        help='Minimum length of the paper to be considered. Default is 2000 characters.')
    parser.add_argument('--max_length', type=int, default=5000,
                        help='Maximum length. Default is 5000 characters.')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Max worker')

    args = parser.parse_args()

    crawler = BBCCrawler()
    my_manager = ProxyManager()

    data = crawler.pipeline(start_date=args.start_date,
                            end_date=args.end_date,
                            max_samples=args.max_samples,
                            min_length=args.min_length,
                            max_length=args.max_length,
                            max_workers=args.max_workers,
                            proxy_manager=my_manager
                            )

    save_json(data, args.file_name)
