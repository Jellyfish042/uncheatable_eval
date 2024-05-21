#!/bin/bash

# Set the start and end date for the data collection
START_DATE="2024-05-01"
END_DATE="2024-05-14"
# Set your GitHub access token
GITHUB_ACCESS_TOKEN=""

echo "Starting data collection from $START_DATE to $END_DATE"

# ArXiv
DATE_RANGE="${START_DATE//-/}to${END_DATE//-/}"

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
GITHUB_DATE_RANGE="${START_DATE//-/}to${END_DATE//-/}"

GITHUB_FILE_NAME="github_cpp_${GITHUB_DATE_RANGE}.json"
GITHUB_LANGUAGE="cpp"
echo "Collecting GitHub data for language $GITHUB_LANGUAGE from $START_DATE to $END_DATE"
python3 github_crawler.py --start_date $START_DATE --end_date $END_DATE --file_name $GITHUB_FILE_NAME --language $GITHUB_LANGUAGE --access_token $GITHUB_ACCESS_TOKEN
if [ $? -eq 0 ]; then
    echo "Data saved to $GITHUB_FILE_NAME"
else
    echo "Failed to collect GitHub data for language $GITHUB_LANGUAGE"
fi

GITHUB_FILE_NAME="github_python_${GITHUB_DATE_RANGE}.json"
GITHUB_LANGUAGE="python"
echo "Collecting GitHub data for language $GITHUB_LANGUAGE from $START_DATE to $END_DATE"
python3 github_crawler.py --start_date $START_DATE --end_date $END_DATE --file_name $GITHUB_FILE_NAME --language $GITHUB_LANGUAGE --access_token $GITHUB_ACCESS_TOKEN
if [ $? -eq 0 ]; then
    echo "Data saved to $GITHUB_FILE_NAME"
else
    echo "Failed to collect GitHub data for language $GITHUB_LANGUAGE"
fi

# Wikipedia
WIKIPEDIA_DATE_RANGE="${START_DATE//-/}to${END_DATE//-/}"
WIKIPEDIA_FILE_NAME="wikipedia_english_${WIKIPEDIA_DATE_RANGE}.json"
echo "Collecting Wikipedia data from $START_DATE to $END_DATE"
python3 wikipedia_crawler.py --start_date $START_DATE --end_date $END_DATE --file_name $WIKIPEDIA_FILE_NAME
if [ $? -eq 0 ]; then
    echo "Data saved to $WIKIPEDIA_FILE_NAME"
else
    echo "Failed to collect Wikipedia data"
fi

# AO3
AO3_DATE_RANGE="${START_DATE//-/}to${END_DATE//-/}"
AO3_FILE_NAME="ao3_${AO3_DATE_RANGE}.json"
echo "Collecting AO3 data from $START_DATE to $END_DATE"
python3 ao3_crawler.py --start_date $START_DATE --end_date $END_DATE --file_name $AO3_FILE_NAME --language english --max_workers 1
if [ $? -eq 0 ]; then
    echo "Data saved to $AO3_FILE_NAME"
else
    echo "Failed to collect AO3 data"
fi

echo "Data collection completed for all sources."
