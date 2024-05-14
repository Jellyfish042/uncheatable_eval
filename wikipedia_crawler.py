import requests
import datetime
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import argparse
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import concurrent.futures
import os
import json


class WikipediaCrawler:
    def __init__(self):
        pass

    @staticmethod
    def create_session():
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        return session

    @staticmethod
    def query(request, session):
        request['action'] = 'query'
        request['format'] = 'json'
        last_continue = {}
        while True:
            req = request.copy()
            req.update(last_continue)
            result = session.get('https://en.wikipedia.org/w/api.php', params=req).json()
            if 'error' in result:
                raise Exception(result['error'])
            if 'warnings' in result:
                print(result['warnings'])
            if 'query' in result:
                yield result['query']
            if 'continue' not in result:
                break
            last_continue = result['continue']

    @staticmethod
    def get_page_content(title, session):
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json"
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ExampleBot/0.1; +http://example.com/bot)'
        }

        try:
            response = session.get(url, headers=headers, params=params)
            data = response.json()

            if 'error' in data:
                return None

            if 'parse' in data:
                html_content = data['parse']['text']['*']
                soup = BeautifulSoup(html_content, 'html.parser')

                if soup.find(class_='redirectMsg'):
                    return None
                references = soup.find('span', {'id': 'References'})
                if references:
                    for sibling in references.parent.find_next_siblings():
                        sibling.decompose()
                    references.parent.decompose()
                for table in soup.find_all('table'):
                    table.decompose()

                return soup.get_text()
            else:
                return None

        except requests.exceptions.SSLError as e:
            print(f"SSL error occurred for title '{title}': {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error occurred for title '{title}': {e}")
            return None

    @staticmethod
    def replace_multiple_newlines(text):
        return re.sub(r'\n+', '\n', text)

    @staticmethod
    def remove_text_in_brackets(text):
        return re.sub(r'\[.*?\]', '', text)

    def process_title(self, title, session, min_length, max_length):
        content = self.get_page_content(title, session)
        if content is not None:
            content = self.replace_multiple_newlines(content)
            content = self.remove_text_in_brackets(content)
            if len(content) > min_length:
                return content[:max_length]
        return None

    def pipeline(self,
                 start_time,
                 end_time,
                 batch_size=500,
                 max_samples=1000,
                 min_length=2000,
                 max_length=5000,
                 num_threads=16):

        request = {
            "list": "recentchanges",
            "rcstart": start_time,
            "rcend": end_time,
            "rcdir": "newer",
            "rctype": "new",
            "rcprop": "title|timestamp",
            "rcnamespace": "0",
            "rclimit": batch_size
        }

        session = self.create_session()

        all_data = []

        pbar = tqdm(total=max_samples)

        # for result in self.query(request, session):
        #     recent_changes = result.get('recentchanges', [])
        #     titles = [change['title'] for change in recent_changes]
        #
        #     for title in titles:
        #         content = self.get_page_content(title, session)
        #         if content is not None:
        #             content = self.replace_multiple_newlines(content)
        #             content = self.remove_text_in_brackets(content)
        #             if len(content) > min_length:
        #                 all_data.append(content[:max_length])
        #                 pbar.update(1)
        #         if len(all_data) >= max_samples:
        #             break
        #
        #     if len(all_data) >= max_samples:
        #         break

        def process_batch(titles):
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(self.process_title, title, session, min_length, max_length): title for title
                           in
                           titles}

                for future in concurrent.futures.as_completed(futures):
                    content = future.result()
                    if content is not None:
                        all_data.append(content)
                        pbar.update(1)
                    if len(all_data) >= max_samples:
                        break

        for result in self.query(request, session):
            recent_changes = result.get('recentchanges', [])
            titles = [change['title'] for change in recent_changes]

            process_batch(titles)

            if len(all_data) >= max_samples:
                break

        pbar.close()
        return all_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str, required=True, help='JSON file name')
    parser.add_argument('--start_date', type=str, required=True,
                        help='The start date (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, required=True,
                        help='The end date (YYYY-MM-DD).')

    # parser.add_argument('--language', type=str, default='english', choices=['english', 'chinese'],
    #                     help='Programming language to filter the repositories.')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of works to crawl. Default is 1000.')
    parser.add_argument('--min_length', type=int, default=2000,
                        help='Minimum length of the files to be considered.')
    parser.add_argument('--max_length', type=int, default=5000,
                        help='Maximum length. Default is 5000 characters.')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Max workers')

    args = parser.parse_args()

    start_time = f"{args.start_date}T00:00:00Z"
    end_time = f"{args.end_date}T23:59:59Z"

    crawler = WikipediaCrawler()
    my_data = crawler.pipeline(start_time=start_time,
                               end_time=end_time,
                               max_samples=args.max_samples,
                               min_length=args.min_length,
                               max_length=args.max_length,
                               num_threads=args.max_workers)

    if not os.path.exists('data'):
        os.makedirs('data')

    file_name = args.file_name.replace('.json', '') + '.json'
    path = os.path.join('data', file_name)

    with open(path, 'w') as f:
        json.dump(my_data, f, ensure_ascii=True, indent=4)
