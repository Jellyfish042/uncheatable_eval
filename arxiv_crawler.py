from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
from helpers import save_json
import os
import tarfile
import tempfile
import requests
from pylatexenc.latex2text import LatexNodes2Text
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import concurrent.futures


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
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            response = session.get(url, timeout=timeout)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                f.write(response.content)

        except requests.exceptions.RequestException as e:
            pass
            # print(f"Error downloading the file: {e}")
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

        if 'main.tex' in [os.path.basename(f) for f in tex_files]:
            return next(f for f in tex_files if os.path.basename(f) == 'main.tex')
        elif len(tex_files) == 1:
            return tex_files[0]
        else:
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

    def extract_text_from_latex(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                latex_content = file.read()

            title_match = re.search(r'\\title{(.*?)}', latex_content, re.DOTALL)
            if title_match:
                title_content = title_match.group(1)
            else:
                # print(f"No title content found in {file_path}")
                title_content = ""

            document_content_match = re.search(r'\\begin{document}(.*?)\\end{document}', latex_content, re.DOTALL)
            if document_content_match:
                document_content = document_content_match.group(1)
            else:
                # print(f"No document content found in {file_path}")
                return ""

            abstract_start_match = re.search(r'\\begin{abstract}', document_content)
            if abstract_start_match:
                abstract_start_index = abstract_start_match.start()
                abstract_to_end_content = document_content[abstract_start_index:]
            else:
                # print(f"No abstract content found in {file_path}")
                return ""

            text_maker = LatexNodes2Text()
            plain_text = text_maker.latex_to_text(abstract_to_end_content)
            plain_text = self.replace_whitespace(plain_text)
            # plain_text = self.clean_extracted_text(plain_text)
            plain_text = self.remove_angle_brackets_content(plain_text)
            plain_text = self.remove_space_before_punctuation(plain_text)
            plain_text = self.replace_whitespace(plain_text)

            if title_content:
                title_content = text_maker.latex_to_text(title_content)
                title_content = self.replace_whitespace(title_content)
                plain_text = title_content + "\n" + plain_text

            return plain_text
        except FileNotFoundError:
            # print(f"Error: The file {file_path} does not exist.")
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
                    # print("Error: Failed to download the file.")
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
        try:
            text = self.download_and_extract_tex_file(src_link)
            if len(text) > min_length:
                return text[:max_length]
        except Exception as e:
            # print(f"Error: {e}")
            return None

    def pipeline_single(self,
                        start_date,
                        end_date,
                        classification,
                        page_size=50,
                        max_samples=1000,
                        min_length=2000,
                        max_length=5000,
                        max_workers=16, ):

        all_data = []

        start_idx = 0

        while True:
            url = f'https://arxiv.org/search/advanced?advanced=1&terms-0-operator=AND&terms-0-term=&terms-0-field=title&classification-{classification}=y&classification-include_cross_list=exclude&date-year=&date-filter_by=date_range&date-from_date={start_date}&date-to_date={end_date}&date-date_type=submitted_date_first&abstracts=hide&size={page_size}&order=-announced_date_first&start={start_idx}'
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            papers = soup.find_all('li', class_='arxiv-result')
            pdf_links = [paper.find('a', string='pdf')['href'] for paper in papers if paper.find('a', string='pdf')]
            src_links = [x.replace('/pdf/', '/src/') for x in pdf_links]

            start_idx += page_size

            for src_link in src_links:
                try:
                    text = self.download_and_extract_tex_file(src_link)
                    if len(text) > min_length:
                        all_data.append(text[:max_length])
                        print(f"Downloaded {len(all_data)} papers")
                except Exception as e:
                    # print(f"Error: {e}")
                    continue
                if len(all_data) >= max_samples:
                    break
            if len(all_data) >= max_samples:
                break

        return all_data

    def pipeline(self,
                 start_date,
                 end_date,
                 classification,
                 page_size=200,
                 max_samples=1000,
                 min_length=2000,
                 max_length=5000,
                 max_workers=16, ):

        assert page_size >= 50, "page_size must be greater than 50"
        assert page_size <= 200, "page_size must be less than 200"

        all_data = []
        start_idx = 0
        pbar = tqdm(total=max_samples)

        while True:
            url = f'https://arxiv.org/search/advanced?advanced=1&terms-0-operator=AND&terms-0-term=&terms-0-field=title&classification-{classification}=y&classification-include_cross_list=exclude&date-year=&date-filter_by=date_range&date-from_date={start_date}&date-to_date={end_date}&date-date_type=submitted_date_first&abstracts=hide&size={page_size}&order=-announced_date_first&start={start_idx}'
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            papers = soup.find_all('li', class_='arxiv-result')
            pdf_links = [paper.find('a', string='pdf')['href'] for paper in papers if paper.find('a', string='pdf')]
            src_links = [x.replace('/pdf/', '/src/') for x in pdf_links]

            start_idx += page_size
            # print(start_idx, len(src_links))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_link, link, min_length, max_length): link for link in src_links}

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_data.append(result[:max_length])
                        pbar.update(1)
                        if len(all_data) >= max_samples:
                            break
            if len(all_data) >= max_samples:
                break

        return all_data[:max_samples]


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
