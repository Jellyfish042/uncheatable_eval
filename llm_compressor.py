import os
import torch
import math
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import struct
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class BitOutputStream:
    def __init__(self, file):
        self.file = file
        self.byte = 0
        self.bit_count = 0

    def write_bit(self, bit):
        self.byte = (self.byte << 1) | bit
        self.bit_count += 1
        if self.bit_count == 8:
            self.file.write(bytes([self.byte]))
            self.byte = 0
            self.bit_count = 0

    def close(self):
        if self.bit_count > 0:
            self.byte <<= 8 - self.bit_count
            self.file.write(bytes([self.byte]))


class BitInputStream:
    def __init__(self, file):
        self.file = file
        self.byte = 0
        self.bit_count = 0

    def read_bit(self):
        if self.bit_count == 0:
            bytes_data = self.file.read(1)
            if not bytes_data:
                return -1
            self.byte = bytes_data[0]
            self.bit_count = 8

        bit = (self.byte >> (self.bit_count - 1)) & 1
        self.bit_count -= 1
        return bit


class ArithmeticEncoder:
    def __init__(self, bit_output, precision=64):
        self.bit_output = bit_output
        self.precision = precision
        self.max_val = (1 << precision) - 1
        self.quarter_val = 1 << (precision - 2)
        self.half_val = 1 << (precision - 1)
        self.three_quarter_val = self.quarter_val * 3
        self.low = 0
        self.high = self.max_val
        self.pending_bits = 0

    def encode_symbol(self, low_count, high_count, total_count):
        range_val = self.high - self.low + 1
        self.high = self.low + (range_val * high_count) // total_count - 1
        self.low = self.low + (range_val * low_count) // total_count

        while True:
            if self.high < self.half_val:
                self.bit_output_bit(0)
            elif self.low >= self.half_val:
                self.bit_output_bit(1)
                self.low -= self.half_val
                self.high -= self.half_val
            elif self.low >= self.quarter_val and self.high < self.three_quarter_val:
                self.pending_bits += 1
                self.low -= self.quarter_val
                self.high -= self.quarter_val
            else:
                break

            self.low <<= 1
            self.high = (self.high << 1) | 1

    def bit_output_bit(self, bit):
        self.bit_output.write_bit(bit)
        while self.pending_bits > 0:
            self.bit_output.write_bit(1 - bit)
            self.pending_bits -= 1

    def finish(self):
        self.pending_bits += 1
        if self.low < self.quarter_val:
            self.bit_output_bit(0)
        else:
            self.bit_output_bit(1)


class ArithmeticDecoder:
    def __init__(self, bit_input, precision=64):
        self.bit_input = bit_input
        self.precision = precision
        self.max_val = (1 << precision) - 1
        self.quarter_val = 1 << (precision - 2)
        self.half_val = 1 << (precision - 1)
        self.three_quarter_val = self.quarter_val * 3
        self.low = 0
        self.high = self.max_val
        self.value = 0

        for _ in range(precision):
            read_val = self.bit_input.read_bit()
            if read_val == -1:
                read_val = 0
            self.value = (self.value << 1) | read_val

    def decode_symbol_find_count(self, total_count):
        range_val = self.high - self.low + 1
        count = ((self.value - self.low + 1) * total_count - 1) // range_val
        return count

    def update_range(self, low_count, high_count, total_count):
        range_val = self.high - self.low + 1
        self.high = self.low + (range_val * high_count) // total_count - 1
        self.low = self.low + (range_val * low_count) // total_count

        while True:
            if self.high < self.half_val:
                pass
            elif self.low >= self.half_val:
                self.value -= self.half_val
                self.low -= self.half_val
                self.high -= self.half_val
            elif self.low >= self.quarter_val and self.high < self.three_quarter_val:
                self.value -= self.quarter_val
                self.low -= self.quarter_val
                self.high -= self.quarter_val
            else:
                break

            self.low <<= 1
            self.high = (self.high << 1) | 1

            bit = self.bit_input.read_bit()
            if bit == -1:
                bit = 0
            self.value = (self.value << 1) | bit


def get_initial_token_id(tokenizer, bos_mode):
    if bos_mode in ["add_default_bos"]:
        bos_token = tokenizer.bos_token_id
    elif bos_mode in ["add_default_eos"]:
        bos_token = tokenizer.eos_token_id
    elif bos_mode in ["add_newline_token"]:
        bos_token = tokenizer.encode("\n", add_special_tokens=False)[0]
    else:
        raise ValueError(f"Invalid BOS mode: {bos_mode}")

    return bos_token


def load_model_and_tokenizer(model_name, model_type="hf", tokenizer_name=None, cache_dir=None, strategy="cuda fp32"):
    if model_type == "hf":
        print(f"Loading HF model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float32,
            cache_dir=cache_dir,
        ).eval()
        return model, tokenizer
    elif model_type == "rwkv7":
        import os

        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "1"
        os.environ["RWKV_V7_ON"] = "1"

        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE

        print(f"Loading RWKV7 model: {model_name}...")
        rwkv_model = RWKV(model=model_name.replace(".pth", ""), strategy=strategy)
        rwkv_pipeline = PIPELINE(rwkv_model, tokenizer_name)
        rwkv_tokenizer = rwkv_pipeline.tokenizer

        return rwkv_model, rwkv_tokenizer
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def compress_file(
    input_path,
    output_path,
    model_name,
    model_type="hf",
    tokenizer_name=None,
    context_window=2048,
    bos_mode="add_default_eos",
    strategy="cuda fp32",
    cache_dir=None,
):
    print("=" * 60)
    print(f"LLM-based Compression - Concept Demonstration ({model_type.upper()})")
    print("WARNING: This is a proof-of-concept implementation.")
    print("Compression/decompression is VERY SLOW and not practical for real use.")
    print("=" * 60)
    print()

    model, tokenizer = load_model_and_tokenizer(model_name, model_type, tokenizer_name, cache_dir, strategy)

    print(f"Reading input file: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text:
        print("Empty file, nothing to compress.")
        return

    if model_type == "hf":
        target_tokens = tokenizer.encode(text, add_special_tokens=False)
    else:  # rwkv7
        tokenized = tokenizer.encode(text)
        if hasattr(tokenized, "ids"):
            target_tokens = tokenized.ids
        else:
            target_tokens = tokenized

    total_tokens = len(target_tokens)
    print(f"Total tokens to compress: {total_tokens}")
    print(f"Context window: {context_window}")

    with open(output_path, "wb") as out_f:
        out_f.write(struct.pack(">I", total_tokens))

        bit_stream = BitOutputStream(out_f)
        encoder = ArithmeticEncoder(bit_stream, precision=64)
        PROB_SCALE = 1 << 48

        start_time = time.time()
        total_nll = 0.0

        context_tokens = []
        rwkv_state = None if model_type == "rwkv7" else None

        if model_type == "hf":
            bos_id = get_initial_token_id(tokenizer, bos_mode)

        for idx in tqdm(range(total_tokens), desc="Compressing"):
            if len(context_tokens) >= context_window:
                context_tokens = []
                if model_type == "rwkv7":
                    rwkv_state = None

            if model_type == "hf":
                if len(context_tokens) == 0:
                    input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=model.device)
                else:
                    input_ids = torch.tensor([context_tokens], dtype=torch.long, device=model.device)

                with torch.no_grad():
                    outputs = model(input_ids, use_cache=False)
                    next_logits = outputs.logits[0, -1, :]
            else:  # rwkv7
                if len(context_tokens) == 0:
                    input_token = 0
                else:
                    input_token = context_tokens[-1]

                logits, rwkv_state = model.forward([input_token], rwkv_state)

                if len(logits.shape) == 2:
                    next_logits = logits[-1]
                else:
                    next_logits = logits

                if isinstance(next_logits, torch.Tensor):
                    next_logits = next_logits.clone().detach().float()
                else:
                    next_logits = torch.tensor(next_logits).float()

            probs = torch.softmax(next_logits.float(), dim=-1)
            counts = (probs * PROB_SCALE).to(torch.long)
            counts = torch.clamp(counts, min=1)

            cdf = torch.cumsum(counts, dim=-1)
            total_count = cdf[-1].item()

            target_id = target_tokens[idx]

            nll = -torch.log(probs[target_id])
            total_nll += nll.item()

            low_val = cdf[target_id - 1].item() if target_id > 0 else 0
            high_val = cdf[target_id].item()

            encoder.encode_symbol(low_val, high_val, total_count)

            context_tokens.append(target_id)

        encoder.finish()
        bit_stream.close()
        end_time = time.time()

    original_size = os.path.getsize(input_path)
    text_byte_size = len(text.encode("utf-8"))

    compressed_size = os.path.getsize(output_path)
    ratio = compressed_size / original_size if original_size > 0 else 0
    duration = end_time - start_time
    speed = total_tokens / duration if duration > 0 else 0

    theoretical_bits = total_nll / math.log(2)
    theoretical_bytes = theoretical_bits / 8
    theoretical_ratio = theoretical_bytes / text_byte_size if text_byte_size > 0 else 0

    print(f"\n--- Compression Results ---")
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {ratio*100:.2f}% (theoretical: {theoretical_ratio*100:.2f}%)")
    print("Compressed file saved to:", output_path)


def decompress_file(
    input_path,
    output_path,
    model_name,
    model_type="hf",
    tokenizer_name=None,
    context_window=2048,
    bos_mode="add_default_eos",
    strategy="cuda fp32",
    cache_dir=None,
):
    print("=" * 60)
    print(f"LLM-based Decompression - Concept Demonstration")
    print("WARNING: This is a proof-of-concept implementation.")
    print("Compression/decompression is VERY SLOW and not practical for real use.")
    print("=" * 60)
    print()

    model, tokenizer = load_model_and_tokenizer(model_name, model_type, tokenizer_name, cache_dir, strategy)

    print(f"Decompressing file: {input_path}")
    with open(input_path, "rb") as in_f:
        data = in_f.read(4)
        if not data:
            print("Empty file")
            return
        total_tokens = struct.unpack(">I", data)[0]
        print(f"Total tokens to decompress: {total_tokens}")
        print(f"Context window: {context_window}")

        bit_stream = BitInputStream(in_f)
        decoder = ArithmeticDecoder(bit_stream, precision=64)
        PROB_SCALE = 1 << 48

        pbar = tqdm(total=total_tokens, desc="Decompressing")
        decoded_tokens = []
        context_tokens = []
        rwkv_state = None if model_type == "rwkv7" else None

        if model_type == "hf":
            bos_id = get_initial_token_id(tokenizer, bos_mode)

        for idx in range(total_tokens):
            if len(context_tokens) >= context_window:
                context_tokens = []
                if model_type == "rwkv7":
                    rwkv_state = None

            if model_type == "hf":
                if len(context_tokens) == 0:
                    input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=model.device)
                else:
                    input_ids = torch.tensor([context_tokens], dtype=torch.long, device=model.device)

                with torch.no_grad():
                    outputs = model(input_ids, use_cache=False)
                    next_logits = outputs.logits[0, -1, :]
            else:  # rwkv7
                if len(context_tokens) == 0:
                    input_token = 0
                else:
                    input_token = context_tokens[-1]

                logits, rwkv_state = model.forward([input_token], rwkv_state)

                if len(logits.shape) == 2:
                    next_logits = logits[-1]
                else:
                    next_logits = logits

                if isinstance(next_logits, torch.Tensor):
                    next_logits = next_logits.clone().detach().float()
                else:
                    next_logits = torch.tensor(next_logits).float()

            probs = torch.softmax(next_logits.float(), dim=0)
            counts = (probs * PROB_SCALE).to(torch.long)
            counts = torch.clamp(counts, min=1)

            cdf = torch.cumsum(counts, dim=0)
            total_count = cdf[-1].item()

            count_val = decoder.decode_symbol_find_count(total_count)
            target_token_index = torch.searchsorted(cdf, count_val, right=True).item()
            target_token_id = target_token_index

            decoded_tokens.append(target_token_id)
            context_tokens.append(target_token_id)

            low_val = cdf[target_token_id - 1].item() if target_token_id > 0 else 0
            high_val = cdf[target_token_id].item()
            decoder.update_range(low_val, high_val, total_count)

            pbar.update(1)

    decoded_text = tokenizer.decode(decoded_tokens)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(decoded_text)

    print(f"\nDecompression finished. Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Text Compression Demo")
    parser.add_argument("input_file", help="Input file path")
    parser.add_argument("output_file", help="Output file path")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base", help="Model name or path")
    parser.add_argument("--model_type", default="hf", choices=["hf", "rwkv7"], help="Model type (hf or rwkv7)")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer name")
    parser.add_argument("--window", type=int, default=2048, help="Context window size")
    parser.add_argument("--bos_mode", default="add_default_eos", help="BOS mode (only for HF models)")
    parser.add_argument("--strategy", default="cuda fp16", help="Strategy for RWKV7 models (e.g., 'cuda fp32', 'cuda fp16')")
    parser.add_argument("--hf_cache_dir", default=None, help="Huggingface cache directory for models")
    parser.add_argument("--task", choices=["compress", "decompress"], default="compress", help="Task to perform")
    args = parser.parse_args()

    if args.model_type == "rwkv7" and args.tokenizer is None:
        raise ValueError("--tokenizer is required for RWKV7 models")

    if args.task == "compress":
        compress_file(
            args.input_file,
            args.output_file,
            args.model,
            args.model_type,
            args.tokenizer,
            args.window,
            args.bos_mode,
            args.strategy,
            args.hf_cache_dir,
        )
    else:
        decompress_file(
            args.input_file,
            args.output_file,
            args.model,
            args.model_type,
            args.tokenizer,
            args.window,
            args.bos_mode,
            args.strategy,
            args.hf_cache_dir,
        )
