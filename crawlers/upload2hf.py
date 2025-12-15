import os
import json
import glob
import argparse
from datasets import Dataset, Features, Value


def load_complex_data(data_dir):

    files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    print(f"Found {len(files)} files...")

    def gen():
        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    row["date"] = str(row["date"])
                    row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
                    yield row

    my_features = Features(
        {
            "content": Value("string"),
            "untruncated_content": Value("string"),
            "category": Value("string"),
            "date": Value("string"),
            "url": Value("string"),
            "metadata": Value("string"),
        }
    )

    dataset = Dataset.from_generator(gen, features=my_features, split="test")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push dataset to Hugging Face Hub.")
    parser.add_argument("--data-dir", default="./data_final", help="Local data directory (default: ./data_final)")
    parser.add_argument("--repo-id", default="uncheatable_eval_data", help="Target Hugging Face repo ID")
    parser.add_argument("--token", required=True, help="Hugging Face access token")
    parser.add_argument("--private", action="store_true", help="Private dataset")
    args = parser.parse_args()

    LOCAL_DIRECTORY = args.data_dir
    HF_REPO_ID = args.repo_id
    HF_TOKEN = args.token
    PRIVATE = args.private

    ds = load_complex_data(LOCAL_DIRECTORY)

    print(f"Size of dataset: {len(ds)}")
    print(ds)

    ds.push_to_hub(HF_REPO_ID, private=PRIVATE, token=HF_TOKEN)
