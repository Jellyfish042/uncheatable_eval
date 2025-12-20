import json
import argparse
from pathlib import Path


def soft_truncate(text, target_length, search_range):
    if len(text) <= target_length:
        return text

    whitespace_chars = set(" \n\r\t\f\v")
    punctuation_chars = set(".,;:!?。、，；：！？")

    start_pos = max(0, target_length - search_range)
    end_pos = min(len(text), target_length + search_range)

    best_pos = None
    for i in range(target_length, start_pos - 1, -1):
        if i < len(text) and (text[i] in whitespace_chars or text[i] in punctuation_chars):
            best_pos = i + 1
            break

    if best_pos is None:
        for i in range(target_length + 1, end_pos):
            if i < len(text) and (text[i] in whitespace_chars or text[i] in punctuation_chars):
                best_pos = i + 1
                break

    if best_pos is not None:
        return text[:best_pos].strip()
    else:
        return text[:target_length].strip()


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_file(input_path, output_path, cut_off_length, max_sample, search_range):
    data = read_jsonl(input_path)
    data = data[:max_sample]
    for sample in data:
        sample["untruncated_content"] = sample["content"]
        sample["content"] = soft_truncate(sample["content"], cut_off_length, search_range)
    save_jsonl(data, output_path)
    return len(data)


def main():
    parser = argparse.ArgumentParser(description="Process JSONL files using soft truncation strategy to truncate text content")
    parser.add_argument("input_path", help="Input JSONL file path or directory path")
    parser.add_argument("output_path", nargs="?", help="Output JSONL file path or directory path (optional, auto-generated if input is a directory)")
    parser.add_argument("--cut-off-length", type=int, default=10000, help="Target truncation length (default: 5000)")
    parser.add_argument("--max-sample", type=int, default=500, help="Maximum number of samples to process (default: 500)")
    parser.add_argument("--search-range", type=int, default=100, help="Soft truncation search range (default: 100)")

    args = parser.parse_args()

    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist")
        return

    if input_path.is_file():
        if args.output_path is None:
            print("Error: Output path is required when input is a file")
            return

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        num_samples = process_file(input_path, output_path, args.cut_off_length, args.max_sample, args.search_range)
        print(f"Processing completed: processed {num_samples} samples, output to {output_path}")

    elif input_path.is_dir():
        jsonl_files = list(input_path.glob("*.jsonl"))

        if not jsonl_files:
            print(f"No JSONL files found in directory '{input_path}'")
            return

        output_dir = Path(args.output_path) if args.output_path else input_path
        output_dir.mkdir(parents=True, exist_ok=True)

        total_samples = 0
        for jsonl_file in jsonl_files:
            output_file = output_dir / f"{jsonl_file.stem}_cutoff{jsonl_file.suffix}"
            num_samples = process_file(jsonl_file, output_file, args.cut_off_length, args.max_sample, args.search_range)
            total_samples += num_samples
            print(f"Processed '{jsonl_file.name}': {num_samples} samples -> '{output_file.name}'")

        print(f"\nProcessing completed: processed {len(jsonl_files)} files, {total_samples} total samples, output to '{output_dir}'")


if __name__ == "__main__":
    main()
