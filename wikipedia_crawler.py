import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import argparse
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import concurrent.futures
from helpers import save_json


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
    def query(request, session, language='english'):
        if language == 'english':
            url = "https://en.wikipedia.org/w/api.php"
        elif language == 'chinese':
            url = "https://zh.wikipedia.org/w/api.php"
        elif language == 'japanese':
            url = "https://ja.wikipedia.org/w/api.php"
        elif language == 'spanish':
            url = "https://es.wikipedia.org/w/api.php"
        elif language == 'german':
            url = "https://de.wikipedia.org/w/api.php"
        elif language == 'french':
            url = "https://fr.wikipedia.org/w/api.php"
        elif language == 'arabic':
            url = "https://ar.wikipedia.org/w/api.php"
        else:
            raise ValueError(f"Language '{language}' is not supported.")
        request['action'] = 'query'
        request['format'] = 'json'
        last_continue = {}
        while True:
            req = request.copy()
            req.update(last_continue)
            result = session.get(url, params=req).json()
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
    def get_page_content(title, session, language='english'):
        params = {
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
        }
        if language == 'english':
            url = "https://en.wikipedia.org/w/api.php"
        elif language == 'chinese':
            url = "https://zh.wikipedia.org/w/api.php"
            params['variant'] = 'zh-cn'
        elif language == 'japanese':
            url = "https://ja.wikipedia.org/w/api.php"
        elif language == 'spanish':
            url = "https://es.wikipedia.org/w/api.php"
        elif language == 'german':
            url = "https://de.wikipedia.org/w/api.php"
        elif language == 'french':
            url = "https://fr.wikipedia.org/w/api.php"
        elif language == 'arabic':
            url = "https://ar.wikipedia.org/w/api.php"
        else:
            raise ValueError(f"Language '{language}' is not supported.")

        try:
            response = session.get(url, params=params)
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
                for math in soup.find_all('math'):
                    math.decompose()
                # for span in soup.find_all('span'):
                #     if 'style' in span.attrs:
                #         span.decompose()

                # Print the structured HTML
                # print("Processed HTML Structure:")
                # print(soup.prettify())
                text = soup.get_text()
                # print('-' * 100)
                # print(text)

                word_list = ['\n参考', '\n注释', '\n注脚', '\n脚注', '\n参考资料', '\n参考文献', '\n参考来源', '\n资料来源', '\n参见', '\n外部链接', '\nReferences', '\n来源', '\n^ ',  # Chinese
                             '\nReferencias', '\n↑ ',  # Spanish
                             '\nWeblinks', '\nEinzelnachweise', '\nLiteratur'  # German
                             '\nRéférences', '\nLiens externes', '\nArticles connexes', '\nArticles connexes', '\nNotes et références', 'Références',  # French
                             '\nReferences', '\n^ ',  # English
                             'المراجع', 'المصادر', 'المصادn', 'المراجع', 'انظر أيضًا', 'مراجع', 'مصادر'  # Arabic
                             ]

                for word in word_list:
                    index = text.find(word)
                    if index != -1:
                        text = text[:index]

                return text
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

    def process_title(self, title, session, min_length, max_length, language):
        content = self.get_page_content(title, session, language)
        if content is not None:
            content = self.replace_multiple_newlines(content)
            content = self.remove_text_in_brackets(content)
            if len(content) > min_length:
                return content[:max_length]
        return None

    def pipeline(self,
                 start_time,
                 end_time,
                 language='english',
                 batch_size=500,
                 max_samples=1000,
                 min_length=1000,
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

        all_data = set()

        pbar = tqdm(total=max_samples)

        def process_batch(titles):
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(self.process_title, title, session, min_length, max_length, language): title for title
                           in
                           titles}

                for future in concurrent.futures.as_completed(futures):
                    content = future.result()
                    if content is not None:
                        all_data.add(content)
                        # print(content)
                        # print('*' * 100)
                        pbar.n = len(all_data)
                        pbar.refresh()
                    if len(all_data) >= max_samples:
                        break

        for result in self.query(request, session, language):
            recent_changes = result.get('recentchanges', [])
            titles = [change['title'] for change in recent_changes]

            process_batch(titles)

            if len(all_data) >= max_samples:
                break

        pbar.close()
        return list(all_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str, required=True, help='JSON file name')
    parser.add_argument('--start_date', type=str, required=True,
                        help='The start date (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, required=True,
                        help='The end date (YYYY-MM-DD).')
    parser.add_argument('--language', type=str, default='english', choices=['english', 'chinese', 'japanese', 'spanish', 'german', 'french', 'arabic'],
                        help='Programming language to filter the repositories.')

    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of works to crawl. Default is 1000.')
    parser.add_argument('--min_length', type=int, default=1000,
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
                               language=args.language,
                               max_samples=args.max_samples,
                               min_length=args.min_length,
                               max_length=args.max_length,
                               num_threads=args.max_workers)

    save_json(my_data, args.file_name)
