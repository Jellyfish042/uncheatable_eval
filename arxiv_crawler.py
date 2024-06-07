from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
from helpers import save_json
import os
import tarfile
import tempfile
import requests
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import concurrent.futures
import time
from pylatexenc.latex2text import LatexNodes2Text


class ArxivCrawler:
    def __init__(self):
        pass

    @staticmethod
    def download_file(url, save_path, retries=3, timeout=40):
        try:
            session = requests.Session()
            retry = Retry(
                total=retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504, 403],
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            head_response = session.head(url, timeout=timeout)
            head_response.raise_for_status()
            content_type = head_response.headers.get('Content-Type')

            # Check if the file type contains 'gzip'
            if 'gzip' not in content_type:
                # print(f"The file type is not gzip, it is: {content_type}")
                return False

            # If the file type is gzip, proceed to download the file
            response = session.get(url, timeout=timeout)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                f.write(response.content)

        except requests.exceptions.RequestException as e:
            # print(f"Error downloading the file: {e}")
            return ""
        return True

    @staticmethod
    def extract_and_list_files(tar_gz_path, extract_path='.'):
        try:
            with tarfile.open(tar_gz_path, "r:gz") as tar:
                tar.extractall(path=extract_path)
                file_names = tar.getnames()

            return file_names

        except tarfile.ReadError:
            # print("Error: The file is not a valid tar.gz archive or is corrupted.")
            return []

    @staticmethod
    def find_tex_file(extracted_files, extract_path):
        tex_files = [os.path.join(extract_path, f) for f in extracted_files if f.endswith('.tex')]
        main_files = ['main.tex', 'paper.tex', 'thesis.tex', 'dissertation.tex']

        for main_file in main_files:
            for tex_file in tex_files:
                if os.path.basename(tex_file) == main_file:
                    return tex_file

        for tex_file in tex_files:
            with open(tex_file, 'r', encoding='utf-8') as file:
                content = file.read()
                if '\\documentclass' in content:
                    return tex_file

        for tex_file in tex_files:
            with open(tex_file, 'r', encoding='utf-8') as file:
                content = file.read()
                if '\\begin{document}' in content:
                    return tex_file

        if len(tex_files) == 1:
            return tex_files[0]

        return None

    @staticmethod
    def replace_whitespace(input_string):
        return re.sub(r'\s+', ' ', input_string).strip()

    @staticmethod
    def remove_angle_brackets_content(input_str):
        result = re.sub(r'<[^>]*>', '', input_str)
        return result

    @staticmethod
    def remove_space_before_punctuation(s):
        s = re.sub(r'\s+([,.!?;:])', r'\1', s)
        return s.strip()

    @staticmethod
    def preprocess_latex(latex_content):
        latex_content = re.sub(r'\\maketitle', '', latex_content)
        latex_content = re.sub(r'\\date{.*?}', '', latex_content, flags=re.DOTALL)
        latex_content = re.sub(r'\\author{.*?}', '', latex_content, flags=re.DOTALL)
        return latex_content

    @staticmethod
    def preprocess_latex_2(latex_content):
        # Replace \href{url}{text} with text (ignoring the URL part)
        latex_content = re.sub(r'\\href{.*?}{(.*?)}', r'\1', latex_content)
        return latex_content

    def extract_text_from_latex(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                latex_content = file.read()

            latex_content = self.preprocess_latex(latex_content)

            title_match = re.search(r'\\title{(.*?)}', latex_content, re.DOTALL)
            if title_match:
                title_content = title_match.group(1)
            else:
                # print(f"No title content found in {file_path}")
                title_content = ""

            abstract_match = re.search(r'\\begin{abstract}(.*?)\\end{abstract}', latex_content, re.DOTALL)
            if abstract_match:
                abstract_content = abstract_match.group(1)
                latex_content = latex_content[:abstract_match.start()] + latex_content[abstract_match.end():]
            else:
                abstract_content = ""
                # print(f"No abstract content found in {file_path}")

            document_content_match = re.search(r'\\begin{document}(.*?)\\end{document}', latex_content, re.DOTALL)
            if document_content_match:
                document_content = document_content_match.group(1)
            else:
                print(f"No document content found in {file_path}")
                return ""

            text_maker = LatexNodes2Text(keep_comments=False)
            # test
            document_content = self.preprocess_latex_2(document_content)
            abstract_content = self.preprocess_latex_2(abstract_content)
            document_plain_text = text_maker.latex_to_text(document_content).strip()
            abstract_plain_text = text_maker.latex_to_text(abstract_content).strip()

            # Clean and process the text
            document_plain_text = self.replace_whitespace(document_plain_text)
            document_plain_text = self.remove_angle_brackets_content(document_plain_text)
            document_plain_text = self.remove_space_before_punctuation(document_plain_text)
            document_plain_text = self.replace_whitespace(document_plain_text)

            if abstract_content:
                abstract_plain_text = self.replace_whitespace(abstract_plain_text)
                document_plain_text = "§ Abstract " + abstract_plain_text + document_plain_text

            if title_content:
                title_content = text_maker.latex_to_text(title_content)
                title_content = self.replace_whitespace(title_content)
                document_plain_text = title_content + "\n" + document_plain_text

            return document_plain_text
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist.")
            return ""
        except Exception as e:
            # print(f"An error occurred while reading the file {file_path}: {e}")
            return ""

    def download_and_extract_tex_file(self, url, temp_dir_path='temp'):
        if not os.path.exists(temp_dir_path):
            os.makedirs(temp_dir_path, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=temp_dir_path) as temp_dir:
                tar_gz_path = os.path.join(temp_dir, 'file.tar.gz')
                if not self.download_file(url, tar_gz_path):
                    return ""

                extract_path = os.path.join(temp_dir, 'extracted_files')
                os.makedirs(extract_path, exist_ok=True)
                file_names = self.extract_and_list_files(tar_gz_path, extract_path)
                if not file_names:
                    # print("Error: No files extracted.")
                    return ""

                tex_file_path = self.find_tex_file(file_names, extract_path)
                if tex_file_path:
                    plain_text = self.extract_text_from_latex(tex_file_path)
                    return plain_text
                else:
                    # print("Error: No .tex file found.")
                    return ""
        except Exception as e:
            # print(f"Unexpected error: {e}")
            return ""

    def process_link(self, src_link, min_length, max_length):
        text = self.download_and_extract_tex_file(src_link)
        if len(text) > min_length:
            return text[:max_length]

    def pipeline_single(self,
                        start_date,
                        end_date,
                        classification,
                        page_size=50,
                        max_samples=1000,
                        min_length=2000,
                        max_length=5000,
                        max_workers=16, ):

        all_data = set()

        start_idx = 0

        while len(all_data) < max_samples:
            url = f'https://arxiv.org/search/advanced?advanced=1&terms-0-operator=AND&terms-0-term=&terms-0-field=title&classification-{classification}=y&classification-include_cross_list=exclude&date-year=&date-filter_by=date_range&date-from_date={start_date}&date-to_date={end_date}&date-date_type=submitted_date_first&abstracts=hide&size={page_size}&order=-announced_date_first&start={start_idx}'
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            papers = soup.find_all('li', class_='arxiv-result')
            pdf_links = [paper.find('a', string='pdf')['href'] for paper in papers if paper.find('a', string='pdf')]
            src_links = [x.replace('/pdf/', '/src/') for x in pdf_links]

            start_idx += page_size

            for src_link in src_links:
                print(src_link)
                try:
                    text = self.download_and_extract_tex_file(src_link)
                    if len(text) > min_length:
                        all_data.add(text[:max_length])
                        print(f"Downloaded {len(all_data)} papers")
                except Exception as e:
                    # print(f"Error: {e}")
                    continue
                if len(all_data) >= max_samples:
                    break

        return all_data[:max_samples]

    def pipeline(self,
                 start_date,
                 end_date,
                 classification,
                 page_size=200,
                 max_samples=1000,
                 min_length=2000,
                 max_length=5000,
                 retries=5,
                 max_workers=16, ):

        assert page_size >= 50, "page_size must be greater than 50"
        assert page_size <= 200, "page_size must be less than 200"

        all_data = set()
        start_idx = 0
        pbar = tqdm(total=max_samples)

        while len(all_data) < max_samples:
            url = f'https://arxiv.org/search/advanced?advanced=1&terms-0-operator=AND&terms-0-term=&terms-0-field=title&classification-{classification}=y&classification-include_cross_list=exclude&date-year=&date-filter_by=date_range&date-from_date={start_date}&date-to_date={end_date}&date-date_type=submitted_date_first&abstracts=hide&size={page_size}&order=-announced_date_first&start={start_idx}'
            for attempt in range(retries):
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    papers = soup.find_all('li', class_='arxiv-result')
                    pdf_links = [paper.find('a', string='pdf')['href'] for paper in papers if
                                 paper.find('a', string='pdf')]
                    src_links = [x.replace('/pdf/', '/src/') for x in pdf_links]

                    start_idx += page_size
                    break
                except (requests.RequestException, requests.Timeout) as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(5)
            else:
                print("All attempts failed. Moving to the next batch.")
                start_idx += page_size
                continue

            if len(src_links) == 0:
                break

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_link, link, min_length, max_length): link for link in src_links}

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_data.add(result[:max_length])
                        pbar.n = len(all_data)
                        pbar.refresh()
                        if len(all_data) >= max_samples:
                            break

        return list(all_data)[:max_samples]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_date', type=str, required=True,
                        help='The start date (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, required=True,
                        help='The end date (YYYY-MM-DD).')
    parser.add_argument('--file_name', type=str, required=True,
                        help='JSON file name')
    parser.add_argument('--classification', type=str, default='computer_science',
                        choices=['computer_science', 'physics', 'mathematics'],
                        help='Default is "cs".')

    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Number of papers to crawl. Default is 1000.')
    parser.add_argument('--page_size', type=int, default=200,
                        help='The number of paper to process in each batch. Default is 100.')
    parser.add_argument('--min_length', type=int, default=2000,
                        help='Minimum length of the paper to be considered. Default is 2000 characters.')
    parser.add_argument('--max_length', type=int, default=5000,
                        help='Maximum length. Default is 5000 characters.')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Max worker')

    args = parser.parse_args()

    crawler = ArxivCrawler()

    my_data = crawler.pipeline(start_date=args.start_date,
                               end_date=args.end_date,
                               classification=args.classification,
                               page_size=args.page_size,
                               max_samples=args.max_samples,
                               min_length=args.min_length,
                               max_length=args.max_length,
                               max_workers=args.max_workers)

    save_json(my_data, args.file_name)
