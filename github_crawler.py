import requests
import base64
import random
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import argparse
import os
from datetime import datetime, timedelta
from threading import Lock
from helpers import save_json


class GitHubCrawler:

    def __init__(self):
        self.lock = Lock()

    def print_rate_limit_status(self, headers):
        with self.lock:
            limit = headers.get('X-RateLimit-Limit')
            remaining = headers.get('X-RateLimit-Remaining')
            reset_timestamp = headers.get('X-RateLimit-Reset')
            if reset_timestamp:
                reset_time = datetime.fromtimestamp(int(reset_timestamp)).strftime('%Y-%m-%d %H:%M:%S')
            else:
                reset_time = "N/A"

            print(f"Rate Limit: {limit}, Remaining: {remaining}, Resets at: {reset_time}")

    def search_github_repos(self, start_date, end_date, language="Python", start_index=1, end_index=100,
                            access_token=None):
        date_list = self.get_date_list(start_date, end_date)
        target_date_idx = start_index // 1000
        real_start_date = date_list[target_date_idx]
        real_end_date = date_list[target_date_idx + 1]

        repo_infos = []
        total_repos = end_index - start_index + 1
        items_fetched = 0
        headers = {}

        if access_token:
            headers['Authorization'] = f'token {access_token}'

        while items_fetched < total_repos:
            per_page = min(100, total_repos - items_fetched)
            page = ((start_index - 1) // 100 + 1) % 10
            query = f"created:{real_start_date}..{real_end_date} language:{language}"
            url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page={per_page}&page={page}"

            try:

                response = requests.get(url, headers=headers)
                # print(url)
                # self.print_rate_limit_status(response.headers)

                if response.status_code == 200:
                    search_results = response.json()
                    for repo in search_results['items']:
                        # Now saving both the repo's URL and its default branch
                        repo_info = {
                            'html_url': repo['html_url'],
                            'default_branch': repo['default_branch']
                        }
                        if repo_info not in repo_infos:
                            repo_infos.append(repo_info)
                            items_fetched += 1
                        if items_fetched >= total_repos:
                            break
                else:
                    print(f"Failed to search repositories. Status code: {response.status_code}")
                    break
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
            start_index += 100

        return repo_infos

    def get_files(self, repo_info, token=None):

        repo_url = repo_info['html_url']
        default_branch = repo_info['default_branch']

        parts = repo_url.split('/')
        owner, repo = parts[-2], parts[-1]

        headers = {'Accept': 'application/vnd.github.v3+json'}
        if token:
            headers['Authorization'] = f'token {token}'

        # url_branch = f"https://api.github.com/repos/{owner}/{repo}/branches/{default_branch}"
        # response_branch = requests.get(url_branch, headers=headers)
        # if response_branch.status_code != 200:
        #     error_message = response_branch.json().get('message', 'No error message provided.')
        #     print(f"Failed to get branch info. Error: {error_message}")
        #     return
        #
        # latest_commit_sha = response_branch.json().get('commit', {}).get('sha')
        # if not latest_commit_sha:
        #     print("Failed to get latest commit SHA. No SHA found in branch info.")
        #     return
        #
        # url_tree = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{latest_commit_sha}?recursive=1"

        url_tree = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"

        response_tree = requests.get(url_tree, headers=headers)
        # self.print_rate_limit_status(response_tree.headers)
        if not response_tree.json().get('tree'):
            error_message = response_tree.json().get('message', 'No error message provided.')
            print(f"Failed to get repository tree. Error: {error_message}")
            return

        files = [item['path'] for item in response_tree.json().get('tree')]
        return files

    def get_content(self, repo, file_path, token=None):
        url = f'https://api.github.com/repos/{repo}/contents/{file_path}'
        headers = {}

        if token:
            headers['Authorization'] = f'token {token}'

        response = requests.get(url, headers=headers)
        # self.print_rate_limit_status(response.headers)

        if response.status_code == 200:
            file_content_base64 = response.json().get('content')
            if file_content_base64:
                file_content_decoded = base64.b64decode(file_content_base64)
                return file_content_decoded.decode('utf-8')
        else:
            print(f"Failed to retrieve file: {file_path}. Status code: {response.status_code}")
            return None

    def process_repo(self, url_info, access_token, suffix, min_length):
        try:
            files = self.get_files(url_info, access_token)
            files = [x for x in files if x.endswith(suffix)]
            if len(files) != 0:
                target_file = random.choice(files)
                content = self.get_content(url_info['html_url'].replace('https://github.com/', ''), target_file,
                                           access_token)
                if content is not None and len(content) > min_length:
                    return content
        except:
            return None

    @staticmethod
    def get_date_list(start_date, end_date):
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        delta = end_date_dt - start_date_dt

        date_list = [(start_date_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]

        return date_list

    def pipeline(self,
                 start_date="2024-04-01",
                 end_date="2024-04-06",
                 language="Python",
                 max_repos=1000,
                 suffix='.py',
                 batch_size=100,
                 min_length=1000,
                 max_length=5000,
                 max_worker=10,
                 access_token=None):

        assert batch_size % 100 == 0

        all_data = []
        global_idx = 0
        pbar = tqdm(total=max_repos)
        # date_list = self.get_date_list(start_date, end_date)
        # max_page = 1000 // batch_size
        while len(all_data) < max_repos:
            start_index = global_idx * batch_size + 1
            end_index = global_idx * batch_size + batch_size
            # print(start_date, end_date, language, start_index, end_index)
            repo_urls = self.search_github_repos(start_date, end_date, language, start_index, end_index, access_token)

            with ThreadPoolExecutor(max_workers=max_worker) as executor:
                future_to_url = {executor.submit(self.process_repo, url, access_token, suffix, min_length): url for url
                                 in
                                 repo_urls}
                for future in as_completed(future_to_url):
                    content = future.result()
                    if content:
                        all_data.append(content)
                        pbar.update(1)
                    if len(all_data) >= max_repos:
                        break

            global_idx += 1
        pbar.close()

        all_data = [x[:max_length] for x in all_data]

        return all_data


def check_date_format(date_str):
    format = "%Y-%m-%d"
    try:
        datetime.strptime(date_str, format)
    except ValueError:
        raise ValueError


LANGUAGE_CONFIG = {
    'python': {'language': 'Python', 'suffix': '.py'},
    'cpp': {'language': 'C%2B%2B', 'suffix': '.cpp'},
    'java': {'language': 'Java', 'suffix': '.java'},
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_date', type=str, required=True,
                        help='The start date (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, required=True,
                        help='The end date (YYYY-MM-DD).')
    parser.add_argument('--file_name', type=str, required=True,
                        help='JSON file name')
    parser.add_argument('--language', type=str, default='python', choices=['python', 'cpp', 'java'],
                        help='Programming language to filter the repositories. Choices are "python" or "cpp". Default is "python".')


    parser.add_argument('--max_repos', type=int, default=1000,
                        help='Maximum number of repositories to crawl. Default is 1000.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The number of repositories to process in each batch. Default is 100.')
    parser.add_argument('--min_length', type=int, default=1000,
                        help='Minimum length of the files to be considered. Default is 1000 characters.')
    parser.add_argument('--max_length', type=int, default=5000,
                        help='Maximum length. Default is 5000 characters.')
    parser.add_argument('--max_worker', type=int, default=20,
                        help='Max worker')
    parser.add_argument('--access_token', type=str, default='', help='Access token for the GitHub API.')

    args = parser.parse_args()

    check_date_format(args.start_date)
    check_date_format(args.end_date)

    crawler = GitHubCrawler()
    all_data = crawler.pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        language=LANGUAGE_CONFIG[args.language]['language'],
        max_repos=args.max_repos,
        suffix=LANGUAGE_CONFIG[args.language]['suffix'],
        batch_size=args.batch_size,
        min_length=args.min_length,
        max_length=args.max_length,
        max_worker=args.max_worker,
        access_token=args.access_token)

    save_json(all_data, args.file_name)
