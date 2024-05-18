#!/bin/bash

# Set the start and end date for the data collection
START_DATE="2024-05-01"
END_DATE="2024-05-14"
# Set your GitHub access token
GITHUB_ACCESS_TOKEN=""

# ArXiv
DATE_RANGE="${START_DATE//-/}to${END_DATE//-/}"

CLASSIFICATION1="computer_science"
FILE_NAME1="arxiv_${CLASSIFICATION1}_${DATE_RANGE}.json"
python3 arxiv_crawler.py --start_date $START_DATE --end_date $END_DATE --classification $CLASSIFICATION1 --file_name $FILE_NAME1

CLASSIFICATION2="physics"
FILE_NAME2="arxiv_${CLASSIFICATION2}_${DATE_RANGE}.json"
python3 arxiv_crawler.py --start_date $START_DATE --end_date $END_DATE --classification $CLASSIFICATION2 --file_name $FILE_NAME2

# GitHub
GITHUB_DATE_RANGE="${START_DATE//-/}to${END_DATE//-/}"

GITHUB_FILE_NAME="github_cpp_${GITHUB_DATE_RANGE}.json"
GITHUB_LANGUAGE="cpp"
python3 github_crawler.py --start_date $GITHUB_START_DATE --end_date $GITHUB_END_DATE --file_name $GITHUB_FILE_NAME --language $GITHUB_LANGUAGE --access_token $GITHUB_ACCESS_TOKEN

GITHUB_FILE_NAME="github_python_${GITHUB_DATE_RANGE}.json"
GITHUB_LANGUAGE="python"
python3 github_crawler.py --start_date $GITHUB_START_DATE --end_date $GITHUB_END_DATE --file_name $GITHUB_FILE_NAME --language $GITHUB_LANGUAGE --access_token $GITHUB_ACCESS_TOKEN

# Wikipedia
WIKIPEDIA_DATE_RANGE="${START_DATE//-/}to${END_DATE//-/}"
WIKIPEDIA_FILE_NAME="wikipedia_english_${WIKIPEDIA_DATE_RANGE}.json"
python3 wikipedia_crawler.py --start_date $WIKIPEDIA_START_DATE --end_date $WIKIPEDIA_END_DATE --file_name $WIKIPEDIA_FILE_NAME
