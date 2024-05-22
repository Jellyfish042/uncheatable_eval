#!/bin/bash

# Set the start and end date for the data collection
START_DATE="2024-05-01"
END_DATE="2024-05-14"
# Set your GitHub access token
GITHUB_ACCESS_TOKEN=""

echo "Starting data collection from $START_DATE to $END_DATE"

DATE_RANGE="${START_DATE//-/}to${END_DATE//-/}"

# ArXiv
CLASSIFICATION1="computer_science"
FILE_NAME1="arxiv_${CLASSIFICATION1}_${DATE_RANGE}.json"
echo "Collecting ArXiv data for classification $CLASSIFICATION1 from $START_DATE to $END_DATE"
python3 arxiv_crawler.py --start_date $START_DATE --end_date $END_DATE --classification $CLASSIFICATION1 --file_name $FILE_NAME1
if [ $? -eq 0 ]; then
    echo "Data saved to $FILE_NAME1"
else
    echo "Failed to collect ArXiv data for classification $CLASSIFICATION1"
fi

CLASSIFICATION2="physics"
FILE_NAME2="arxiv_${CLASSIFICATION2}_${DATE_RANGE}.json"
echo "Collecting ArXiv data for classification $CLASSIFICATION2 from $START_DATE to $END_DATE"
python3 arxiv_crawler.py --start_date $START_DATE --end_date $END_DATE --classification $CLASSIFICATION2 --file_name $FILE_NAME2
if [ $? -eq 0 ]; then
    echo "Data saved to $FILE_NAME2"
else
    echo "Failed to collect ArXiv data for classification $CLASSIFICATION2"
fi

# GitHub
GITHUB_FILE_NAME="github_cpp_${DATE_RANGE}.json"
GITHUB_LANGUAGE="cpp"
echo "Collecting GitHub data for language $GITHUB_LANGUAGE from $START_DATE to $END_DATE"
python3 github_crawler.py --start_date $START_DATE --end_date $END_DATE --file_name $GITHUB_FILE_NAME --language $GITHUB_LANGUAGE --access_token $GITHUB_ACCESS_TOKEN
if [ $? -eq 0 ]; then
    echo "Data saved to $GITHUB_FILE_NAME"
else
    echo "Failed to collect GitHub data for language $GITHUB_LANGUAGE"
fi

GITHUB_FILE_NAME="github_python_${DATE_RANGE}.json"
GITHUB_LANGUAGE="python"
echo "Collecting GitHub data for language $GITHUB_LANGUAGE from $START_DATE to $END_DATE"
python3 github_crawler.py --start_date $START_DATE --end_date $END_DATE --file_name $GITHUB_FILE_NAME --language $GITHUB_LANGUAGE --access_token $GITHUB_ACCESS_TOKEN
if [ $? -eq 0 ]; then
    echo "Data saved to $GITHUB_FILE_NAME"
else
    echo "Failed to collect GitHub data for language $GITHUB_LANGUAGE"
fi

# Wikipedia
WIKIPEDIA_FILE_NAME="wikipedia_english_${DATE_RANGE}.json"
echo "Collecting Wikipedia data from $START_DATE to $END_DATE"
python3 wikipedia_crawler.py --start_date $START_DATE --end_date $END_DATE --file_name $WIKIPEDIA_FILE_NAME
if [ $? -eq 0 ]; then
    echo "Data saved to $WIKIPEDIA_FILE_NAME"
else
    echo "Failed to collect Wikipedia data"
fi

# AO3
LANGUAGE="chinese"
AO3_FILE_NAME="ao3_${LANGUAGE}_${DATE_RANGE}.json"
echo "Collecting AO3 data from $START_DATE to $END_DATE in $LANGUAGE"
echo "AO3 has a strict rate limit (20 requests per minute), please implement your own proxy strategy in proxy.py, then set max_workers to larger value, or wait for a longer period."
python3 ao3_crawler.py --start_date $START_DATE --end_date $END_DATE --file_name $AO3_FILE_NAME --language $LANGUAGE --max_workers 16
if [ $? -eq 0 ]; then
    echo "Data saved to $AO3_FILE_NAME"
else
    echo "Failed to collect AO3 data"
fi

# BBC News
BBC_FILE_NAME="bbc_news_${DATE_RANGE}.json"
echo "Collecting BBC News data from $START_DATE to $END_DATE"
python3 bbc_crawler.py --start_date $START_DATE --end_date $END_DATE --file_name $BBC_FILE_NAME
if [ $? -eq 0 ]; then
    echo "Data saved to $BBC_FILE_NAME"
else
    echo "Failed to collect BBC News data"
fi

echo "Data collection completed for all sources."