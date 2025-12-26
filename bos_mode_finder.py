import tempfile
from pathlib import Path
from evaluator import EvaluationConfig, Evaluator
from transformers import AutoTokenizer
import json
import os
import glob
import argparse


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_best_bos_config(model_name_or_path, tokenizer_name_or_path, cache_dir, model_type):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir, trust_remote_code=True)

    print(f"BOS token: {tokenizer.bos_token_id}")
    print(f"EOS token: {tokenizer.eos_token_id}")
    token_id = tokenizer.encode("\n")[0]
    print(f"\\n token: {token_id}")

    print("Checking tokenizer behavior...")

    dataset = read_jsonl("support/uncheatable_eval_dev.jsonl")
    texts = [item["content"] for item in dataset]
    first_token_set = set()
    for sample in texts:
        inputs = tokenizer(sample, return_tensors="pt", add_special_tokens=False)
        first_token_set.add(inputs["input_ids"][0, 0].item())

    mode_to_try = []
    if len(first_token_set) > 1:
        print("This tokenizer will NOT add BOS token when you pass add_special_tokens=False")
        if tokenizer.bos_token_id is not None:
            mode_to_try.append("add_default_bos")
        if tokenizer.eos_token_id is not None:
            mode_to_try.append("add_default_eos")
        mode_to_try.append("add_newline_token")
    else:
        print("This tokenizer will add BOS token when you pass add_special_tokens=False")
        if tokenizer.bos_token_id is not None:
            mode_to_try.append("replace_with_bos")
        if tokenizer.eos_token_id is not None:
            mode_to_try.append("replace_with_eos")
        mode_to_try.append("replace_with_newline_token")

    print(f"Trying the following BOS modes: {mode_to_try}")

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:

        evaluator = Evaluator()

        for mode in mode_to_try:
            config = EvaluationConfig(
                model_name_or_path=model_name_or_path,
                tokenizer_name=tokenizer_name_or_path,
                model_type=model_type,
                data=["support/uncheatable_eval_dev.jsonl"],
                bos_mode=mode,
                log_path=temp_dir,
                cache=cache_dir,
            )
            evaluator.evaluate(config)

        jsonl_files = glob.glob(os.path.join(temp_dir, "*.json"))
        min_compression_ratio = float("inf")
        for file_path in jsonl_files:
            results = read_json(file_path)
            bos_mode = results["bos_mode"]
            compression_ratio = results["compression_rate"]
            if compression_ratio < min_compression_ratio:
                min_compression_ratio = compression_ratio
                best_bos_mode = bos_mode
            print(f"BOS mode: {bos_mode}, Compression ratio: {compression_ratio}")

    print(f"For Model: {model_name_or_path} and Tokenizer: {tokenizer_name_or_path}, you should use BOS mode: {best_bos_mode}")

    return best_bos_mode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best BOS mode configuration for a model")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of the model")
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="Path or name of the tokenizer")
    parser.add_argument("--model_type", type=str, default="hf", choices=["hf", "mamba", "mistral"], help="Type of the model (default: hf)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for models and tokenizers (default: None)")

    args = parser.parse_args()

    find_best_bos_config(args.model_name_or_path, args.tokenizer_name_or_path, args.cache_dir, args.model_type)
