import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import unicodedata

# Tokenizer models for multi-tokenizer truncation
TOKENIZER_MODELS = [
    "gpt2",
    "google/gemma-3-270m",
    "Qwen/Qwen3-0.6B-Base",
    "meta-llama/Llama-2-7b-hf",
    "EleutherAI/gpt-neox-20b",
    "HuggingFaceTB/SmolLM2-135M",
    "tiiuae/Falcon-H1-0.5B-Base",
    "mistralai/Mistral-7B-v0.1",
]


def nfc_normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def truncate_by_bytes(text: str, max_bytes: int, encoding: str = "utf-8") -> str:
    encoded = text.encode(encoding)
    if len(encoded) <= max_bytes:
        return text
    sliced_bytes = encoded[:max_bytes]
    return sliced_bytes.decode(encoding, errors="ignore")


def soft_truncate(text, target_length, search_range):

    if target_length is None:
        return text.strip()

    if len(text) <= target_length:
        return text.strip()

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


def get_max_token_count(text, tokenizers):
    """Get the maximum token count across all tokenizers."""
    max_count = 0
    for tokenizer in tokenizers:
        tokens = tokenizer.encode(text)
        max_count = max(max_count, len(tokens))
    return max_count


def truncate_with_token_limit(text, cut_off_length, search_range, max_tokens, tokenizers):
    """
    Truncate text using soft truncation and ensure token count doesn't exceed max_tokens
    for ALL tokenizers. Uses the maximum token count across all tokenizers to ensure
    the sequence length is within limit regardless of which tokenizer is used.
    """
    if max_tokens is None:
        # No token limit, just do regular soft truncation
        return soft_truncate(text, cut_off_length, search_range)

    # First soft truncate
    truncated = soft_truncate(text, cut_off_length, search_range)

    # Check token count across all tokenizers
    token_count = get_max_token_count(truncated, tokenizers)

    # If within limit, return
    if token_count <= max_tokens:
        return truncated

    # Otherwise, iteratively reduce target_length
    current_target = len(truncated)

    while token_count > max_tokens and current_target > 0:
        reduction_ratio = max_tokens / token_count
        current_target = int(current_target * reduction_ratio * 0.95)
        current_target = max(1, current_target)

        truncated = soft_truncate(text, current_target, search_range)
        token_count = get_max_token_count(truncated, tokenizers)

    return truncated


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_file(input_path, output_path, cut_off_length, max_sample, search_range, max_tokens, max_bytes, tokenizers, tokenizer_names):
    data = read_jsonl(input_path)
    data = data[:max_sample]
    for sample in data:
        sample["content"] = nfc_normalize(sample["content"])
        sample["untruncated_content"] = sample["content"]
        if max_bytes is not None:
            sample["content"] = truncate_by_bytes(sample["content"], max_bytes)
        sample["content"] = truncate_with_token_limit(sample["content"], cut_off_length, search_range, max_tokens, tokenizers)
    save_jsonl(data, output_path)

    # Collect statistics per tokenizer
    token_counts_per_tokenizer = {name: [] for name in tokenizer_names}
    char_counts = []
    byte_counts = []
    for sample in data:
        content = sample["content"]
        for tokenizer, name in zip(tokenizers, tokenizer_names):
            tokens = tokenizer.encode(content)
            token_counts_per_tokenizer[name].append(len(tokens))
        char_counts.append(len(content))
        byte_counts.append(len(content.encode("utf-8")))

    # Print statistics
    if char_counts:
        print(f"    ├── Samples: {len(data)}")
        for name in tokenizer_names:
            token_counts = token_counts_per_tokenizer[name]
            avg_tokens = sum(token_counts) / len(token_counts)
            max_tokens_stat = max(token_counts)
            min_tokens_stat = min(token_counts)
            short_name = name.split("/")[-1] if "/" in name else name
            print(f"    ├── Tokens ({short_name}): avg={avg_tokens:.1f}, min={min_tokens_stat}, max={max_tokens_stat}")
        avg_chars = sum(char_counts) / len(char_counts)
        max_chars = max(char_counts)
        min_chars = min(char_counts)
        avg_bytes = sum(byte_counts) / len(byte_counts)
        max_bytes_stat = max(byte_counts)
        min_bytes_stat = min(byte_counts)
        print(f"    ├── Chars: avg={avg_chars:.1f}, min={min_chars}, max={max_chars}")
        print(f"    └── Bytes: avg={avg_bytes:.1f}, min={min_bytes_stat}, max={max_bytes_stat}")

    return len(data)


def main():
    parser = argparse.ArgumentParser(description="Process JSONL files using soft truncation strategy to truncate text content")
    parser.add_argument("input_path", help="Input JSONL file path or directory path")
    parser.add_argument("output_path", nargs="?", help="Output JSONL file path or directory path (optional, auto-generated if input is a directory)")
    parser.add_argument(
        "--cut-off-length",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=None,
        help="Target truncation length, or 'none' to disable (default: 10000)",
    )
    parser.add_argument(
        "--max-sample",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=500,
        help="Maximum number of samples to process, or 'none' to disable (default: 500)",
    )
    parser.add_argument("--search-range", type=int, default=100, help="Soft truncation search range (default: 100)")
    parser.add_argument(
        "--max-tokens",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=3584,
        help="Maximum token count across all tokenizers, or 'none' to disable (default: 3800)",
    )
    parser.add_argument(
        "--max-bytes",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=None,
        help="Maximum byte count (UTF-8), or 'none' to disable (default: none)",
    )

    args = parser.parse_args()

    # Load all tokenizers
    print("Loading tokenizers...")
    tokenizers = []
    tokenizer_names = TOKENIZER_MODELS
    for model_name in tokenizer_names:
        print(f"  Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizers.append(tokenizer)
    print(f"  Loaded {len(tokenizers)} tokenizers")

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

        print(f"\n{'='*60}")
        print(f"Processing: {input_path.name}")
        print(f"{'='*60}")
        print(f"  Output: {output_path}")
        num_samples = process_file(
            input_path,
            output_path,
            args.cut_off_length,
            args.max_sample,
            args.search_range,
            args.max_tokens,
            args.max_bytes,
            tokenizers,
            tokenizer_names,
        )
        print(f"{'='*60}")
        print(f"Done: {num_samples} samples processed")
        print(f"{'='*60}\n")

    elif input_path.is_dir():
        jsonl_files = list(input_path.glob("*.jsonl"))

        if not jsonl_files:
            print(f"No JSONL files found in directory '{input_path}'")
            return

        output_dir = Path(args.output_path) if args.output_path else input_path
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Batch Processing: {input_path}")
        print(f"{'='*60}")
        print(f"  Files found: {len(jsonl_files)}")
        print(f"  Output dir:  {output_dir}")
        print(f"  Settings:    cut_off={args.cut_off_length}, max_tokens={args.max_tokens}, max_bytes={args.max_bytes}, max_sample={args.max_sample}")
        print(f"{'='*60}")

        total_samples = 0
        for i, jsonl_file in enumerate(jsonl_files, 1):
            output_file = output_dir / f"{jsonl_file.stem}_cutoff{jsonl_file.suffix}"
            print(f"\n[{i}/{len(jsonl_files)}] {jsonl_file.name} -> {output_file.name}")
            num_samples = process_file(
                jsonl_file,
                output_file,
                args.cut_off_length,
                args.max_sample,
                args.search_range,
                args.max_tokens,
                args.max_bytes,
                tokenizers,
                tokenizer_names,
            )
            total_samples += num_samples

        print(f"\n{'='*60}")
        print(f"Summary")
        print(f"{'='*60}")
        print(f"  Files processed:  {len(jsonl_files)}")
        print(f"  Total samples:    {total_samples}")
        print(f"  Output directory: {output_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
