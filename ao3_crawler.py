import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from helpers import save_json
from proxy import ProxyManager
from datetime import datetime, timedelta
from lxml import html
import time
import re
from requests.exceptions import RequestException


class AO3Crawler:
    def __init__(self):
        pass

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
        # Convert input strings to date objects
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        # Get today's date
        today = datetime.today().date()

        # Check if either date is later than or equal to today
        if start_date >= today or end_date >= today:
            return "Error: Dates must be earlier than today."

        # Calculate the difference in days from today to the start and end dates
        days_from_today_to_end = (today - end_date).days
        days_from_today_to_start = (today - start_date).days

        # Format the result string
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
                response = requests.get(url, proxies=proxies)

                if response.status_code == 200:
                    tree = html.fromstring(response.content)
                    for xpath in xpaths:
                        paragraphs = tree.xpath(xpath)
                        if paragraphs:
                            result = [paragraph.text_content().replace('\xa0', '\n') for paragraph in paragraphs]
                            result = ''.join(result)
                            result = re.sub(r'\n+', '\n', result)
                            return result.strip()

                    # print(f"No paragraphs found with any of the given XPaths. Attempt {attempt + 1} of {retries}.")
                else:
                    pass
                    # print(f"Failed to retrieve the webpage. Status code: {response.status_code}. Attempt {attempt + 1} of {retries}.")

            except requests.RequestException as e:
                pass
                # print(f"Request failed: {e}. Attempt {attempt + 1} of {retries}.")

            time.sleep(delay)

        # print("All attempts failed.")
        return []

    @staticmethod
    def fetch_url_with_retries(url, max_retries=5, delay=2, verify=True, proxies=None):
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url, verify=verify, proxies=proxies)
                return response
            except RequestException as e:
                retries += 1
                # print(f"Attempt {retries} failed with error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
        raise Exception(f"Failed to fetch URL after {max_retries} attempts")

    def pipeline(self,
                 start_date,
                 end_date,
                 language_id='1',
                 num_works=1000,
                 max_workers=1,
                 max_page=100,
                 min_char=2000,
                 max_char=5000,
                 sleep_time=3,
                 manager=None):

        set_time = self.calculate_dates(start_date, end_date)

        url_template = 'https://archiveofourown.org/works/search?commit=Search&page=<PAGE>&work_search[language_id]=<LANGUAGE_ID>&work_search%5Brevised_at=<TIME>&work_search[single_chapter]=0&work_search[sort_column]=created_at&work_search[sort_direction]=desc'

        pbar = tqdm(total=num_works)
        all_works = set()

        def fetch_and_process_work(work_id, proxies=None):
            try:
                final_text = self.get_work(work_id, proxies=proxies)[:max_char]
                if len(final_text) > min_char:
                    return final_text
                return None
            except:
                print(f"Failed to fetch work {work_id}")
                return None

        if max_workers > 1:
            for page in range(1, max_page + 1):
                url = url_template.replace('<PAGE>', str(page)).replace('<LANGUAGE_ID>', str(language_id)).replace(
                    '<TIME>', set_time)
                proxy = manager.get_random_proxy()
                response = self.fetch_url_with_retries(url, proxies=proxy)
                if response.status_code == 200:
                    content = response.text
                    work_ids = self.extract_filtered_work_ids(content)

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        proxy = manager.get_random_proxy()
                        futures = [executor.submit(fetch_and_process_work, work_id, proxy) for work_id in work_ids]
                        for future in as_completed(futures):
                            result = future.result()
                            if result:
                                all_works.add(result)
                                pbar.n = len(all_works)
                                pbar.refresh()
                                if len(all_works) >= num_works:
                                    return list(all_works)[:num_works]
                else:
                    pass
                    # print(f"Failed to retrieve content, status code: {response.status_code}")

            return list(all_works)[:num_works]

        else:
            for page in range(max_page):
                url = url_template.replace('<PAGE>', str(page)).replace('<LANGUAGE_ID>', str(language_id)).replace(
                    '<TIME>', set_time)

                proxy = manager.get_random_proxy()
                response = self.fetch_url_with_retries(url, proxies=proxy)

                if response.status_code == 200:
                    content = response.text

                    work_ids = self.extract_filtered_work_ids(content)
                    # print(work_ids)

                    for work_id in work_ids:
                        try:
                            final_text = self.get_work(work_id)[:max_char]
                        except:
                            continue

                        time.sleep(sleep_time)

                        if len(final_text) > min_char:
                            all_works.add(final_text)
                            pbar.n = len(all_works)
                            pbar.refresh()

                            if len(all_works) >= num_works:
                                break

                else:
                    # print(f"Failed to retrieve content, status code: {response.status_code}")
                    pass

                if len(all_works) >= num_works:
                    break

            return list(all_works)[:num_works]


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
