import importlib.metadata
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Any
import subprocess
import sys
from datetime import datetime
from packaging import version
import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


def get_decoded(tokenizer, text):
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    return [[u2b[char] for char in token] for token in tokens]


@dataclass
class EvaluationConfig:
    model_name_or_path: str
    tokenizer_name: str
    model_type: str
    data: list

    model_args: Dict[Any, Any] = field(default_factory=dict)  # other arguments that can be passed to the Hugging Face AutoModelForCausalLM
    tokenizer_args: Dict[Any, Any] = field(default_factory=dict)  # other arguments that can be passed to the Hugging Face AutoTokenizer

    requirements: list = field(default_factory=list)  # list of packages, will be installed automatically

    bos_mode: str = (
        "add_default_bos"  # how to handle the bos token, only work for hf models, use bos_mode_finder.py to find the best bos mode unless you know what you are doing
    )
    log_path: str = "./logs/"  # path to save the evaluation results
    cache: str = None  # cache directory for the models
    enable_chunking: bool = True  # whether to enable chunking
    chunk_size: int = 4000  # input tokens will be split into chunks of this size
    batch_size: int = 1  # batch size for inference, batch size > 1 is buggy and not supported yet
    track_byte_wise_data: bool = False  # whether to track byte-wise data

    def __post_init__(self):

        assert self.batch_size == 1, "Batch size > 1 is buggy and not supported yet"
        assert self.chunk_size > 2, "Chunk size must be greater than 2"
        if self.track_byte_wise_data:
            assert not self.enable_chunking, "Chunking must be disabled to track byte-wise data"

        assert self.bos_mode in [
            "add_default_bos",
            "add_default_eos",
            "add_newline_token",
            "replace_with_bos",
            "replace_with_eos",
            "replace_with_newline_token",
        ], "Invalid bos mode"

        default_model_args = {"device_map": "auto", "trust_remote_code": True}
        self.model_args = {**default_model_args, **self.model_args}

        default_tokenizer_args = {"trust_remote_code": True}
        self.tokenizer_args = {**default_tokenizer_args, **self.tokenizer_args}

        if not os.path.exists(self.model_name_or_path):
            self.original_model_name_or_path = self.model_name_or_path
            if ".pth" in self.model_name_or_path and "rwkv" in self.model_name_or_path.lower() and self.cache:
                self.model_name_or_path = os.path.join(self.cache, self.model_name_or_path.split("/")[-1])

        if not self.enable_chunking:
            self.chunk_size = int(1e30)


class Evaluator:
    def __init__(self):
        pass

    @staticmethod
    def load_list_from_json(file_path):
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as file:
                data = [line for line in file]
            return [json.loads(line)["content"] for line in data]
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)

    @staticmethod
    def default_serializer(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)

    @staticmethod
    def install_requirements(requirements: list):
        """
        Installs or upgrades packages and reloads them if already imported.

        :param requirements: List of packages with potential version specifiers.
        """
        for requirement in requirements:
            package_info = (
                requirement.split("==")
                if "==" in requirement
                else (requirement.split(">=") if ">=" in requirement else (requirement.split("<=") if "<=" in requirement else [requirement]))
            )
            package_name = package_info[0]
            required_version_spec = requirement[len(package_name) :]

            try:
                # Check if the package is already installed
                dist = importlib.metadata.distribution(package_name)
                installed_version = version.parse(dist.version)
                needs_reload = False

                if required_version_spec:
                    # Extract the operator and the version from requirement
                    operator = required_version_spec[:2] if required_version_spec[1] in ["=", ">"] else required_version_spec[0]
                    required_version = version.parse(required_version_spec.lstrip(operator))

                    # Version comparison based on the operator
                    if (
                        (operator == "==" and installed_version == required_version)
                        or (operator == ">=" and installed_version >= required_version)
                        or (operator == "<=" and installed_version <= required_version)
                    ):
                        print(f"Package {package_name} already installed and meets the requirement {requirement}.")
                    else:
                        needs_reload = True
                        print(f"Package {package_name} version {installed_version} does not meet the requirement {requirement}, upgrading...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}{required_version_spec}"])
                else:
                    print(f"Package {package_name} is already installed.")

                if needs_reload:
                    # Reload the package if it was already imported
                    if package_name in sys.modules:
                        importlib.reload(sys.modules[package_name])
                        print(f"Reloaded {package_name}")
            except importlib.metadata.PackageNotFoundError:
                # Package is not installed, install it with the specified version
                print(f"Package {package_name} is not installed, installing {requirement}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Error installing or upgrading package {package_name}: {e}")

    @staticmethod
    def print_model_parameters_in_billions(model):
        total_params = sum(p.numel() for p in model.parameters())

        total_params_billion = total_params / 1e9

        print(f"Model parameters: {total_params_billion:.3f} billion")

    @staticmethod
    def print_rwkv7_parameters_in_billions(rwkv_model):
        total_params = 0
        for param in rwkv_model.z.values():
            total_params += param.numel()
        print(f"Model parameters: {total_params / 1e9:.3f} billion")

    @staticmethod
    def print_rwkv_parameters_in_billions(rwkv_model):
        total_params = 0
        for param in rwkv_model.w.values():
            total_params += param.numel()
        print(f"Model parameters: {total_params / 1e9:.3f} billion")

    def load_hf_model(self, config: EvaluationConfig):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        hf_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, cache_dir=config.cache, **config.tokenizer_args)
        # if config.track_byte_wise_data:
        #     hf_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, cache_dir=config.cache, **config.tokenizer_args, use_fast=False)
        # else:
        #     hf_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, cache_dir=config.cache, **config.tokenizer_args)
        hf_model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, cache_dir=config.cache, **config.model_args).eval()

        self.print_model_parameters_in_billions(hf_model)

        return hf_model, hf_tokenizer

    def load_rwkv(self, config: EvaluationConfig):
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "1"

        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE

        if config.model_args.get("strategy") is not None:
            rwkv_model = RWKV(model=config.model_name_or_path, strategy=config.model_args.get("strategy"))
        else:
            rwkv_model = RWKV(model=config.model_name_or_path, strategy="cuda fp16")
        rwkv_pipeline = PIPELINE(rwkv_model, config.tokenizer_name)
        rwkv_tokenizer = rwkv_pipeline.tokenizer

        self.print_rwkv_parameters_in_billions(rwkv_model)

        return rwkv_model, rwkv_tokenizer

    def load_rwkv7(self, config: EvaluationConfig):
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "1"
        os.environ["RWKV_V7_ON"] = "1"

        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE

        if config.model_args.get("strategy") is not None:
            rwkv_model = RWKV(model=config.model_name_or_path.replace(".pth", ""), strategy=config.model_args.get("strategy"))
        else:
            rwkv_model = RWKV(model=config.model_name_or_path.replace(".pth", ""), strategy="cuda fp16")
        rwkv_pipeline = PIPELINE(rwkv_model, config.tokenizer_name)
        rwkv_tokenizer = rwkv_pipeline.tokenizer

        self.print_rwkv7_parameters_in_billions(rwkv_model)

        return rwkv_model, rwkv_tokenizer

    def load_mamba(self, config: EvaluationConfig):
        # state-spaces/mamba-2.8b-slimpj
        # pip install mamba-ssm
        # pip install causal-conv1d>=1.2.0
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        model = MambaLMHeadModel.from_pretrained(config.model_name_or_path, device="cuda", dtype=torch.float16)
        model.device = torch.device("cuda")

        self.print_model_parameters_in_billions(model)

        return model, tokenizer

    @staticmethod
    def calculate_log_sum(logits, target_token_ids, reduction="none"):
        return F.cross_entropy(logits[:-1], target_token_ids[1:], reduction=reduction)

    @staticmethod
    def count_rwkv_parameters_in_billions(rwkv_model):
        total_params = 0
        if hasattr(rwkv_model, "z"):
            for param in rwkv_model.z.values():
                total_params += param.numel()
        if hasattr(rwkv_model, "w"):
            for param in rwkv_model.w.values():
                total_params += param.numel()
        return total_params / 1e9

    @staticmethod
    def count_model_parameters_in_billions(model):
        total_params = sum(p.numel() for p in model.parameters())

        total_params_billion = total_params / 1e9

        return total_params_billion

    @staticmethod
    def get_string_byte_size(input_string):
        # Encode the string into bytes using UTF-8 encoding
        byte_array = input_string.encode("utf-8")
        # Calculate the length of the byte array
        byte_size = len(byte_array)
        return byte_size

    def eval_rwkv(self, model, tokenizer, texts, chunk_size, bos_mode, track_byte_wise_data):
        rwkv_test_data = []
        rwkv_token_length_list = []
        char_count = []

        if track_byte_wise_data:
            max_byte_size = max(len(text.encode("utf-8")) for text in texts)
            byte_wise_loss_sum = [0.0 for _ in range(max_byte_size)]
            byte_wise_counts = [0 for _ in range(max_byte_size)]

        for idx, sample in tqdm(enumerate(texts), total=len(texts)):

            char_count.append(len(sample))

            tokenized = tokenizer.encode(sample)
            if hasattr(tokenized, "ids"):
                input_seq = tokenized.ids  # RWKV v4pile
            else:
                input_seq = tokenized  # RWKV world

            input_length = len(input_seq)

            neg_log_prob_temp = 0
            for begin in range(0, input_length, chunk_size):

                input_chunk = [0] + input_seq[begin : begin + chunk_size]

                logit = model.forward(input_chunk, None, full_output=True)[0]

                if len(input_chunk) == 1:
                    logit = logit.unsqueeze(0)

                loss = self.calculate_log_sum(logit, torch.tensor(input_chunk).cuda())

                neg_log_prob_temp += loss.sum().item()

            rwkv_token_length_list.append(input_length)
            rwkv_test_data.append(neg_log_prob_temp)

            byte_index = 0
            if track_byte_wise_data:  # track byte-wise data is not compatible with chunking, so here we will get the whole sequence's loss
                token_bytes = [tokenizer.decodeBytes([token]) for token in input_chunk[1:]]
                for l, byte_values in zip(loss.tolist(), token_bytes):
                    per_byte_loss = l / len(byte_values)
                    for _ in range(len(byte_values)):
                        byte_wise_loss_sum[byte_index] += per_byte_loss
                        byte_wise_counts[byte_index] += 1
                        byte_index += 1

        avg_byte_wise_loss = []
        if track_byte_wise_data:
            for s, c in zip(byte_wise_loss_sum, byte_wise_counts):
                if c > 0:
                    avg_byte_wise_loss.append(s / c)
                else:
                    break

        data_dict = {
            "neg_log_prob_sum": sum(rwkv_test_data) / len(rwkv_test_data),
            "avg tokens": sum(rwkv_token_length_list) / len(rwkv_token_length_list),
            "avg character count": sum(char_count) / len(char_count),
            "parameters count": self.count_rwkv_parameters_in_billions(model),
            "avg bytes": sum([self.get_string_byte_size(text) for text in texts]) / len(texts),
            "sample_count": len(texts),
        }

        if track_byte_wise_data:
            data_dict["byte_wise_loss"] = avg_byte_wise_loss
            data_dict["byte_wise_counts"] = byte_wise_counts
            factor = (1.0 / math.log(2.0)) * 0.125 * 100.0
            data_dict["byte_wise_compression_rate"] = [loss_val * factor for loss_val in avg_byte_wise_loss]

        return data_dict

    @torch.no_grad()
    def eval_hf_model(self, model, tokenizer, texts, chunk_size, bos_mode, track_byte_wise_data):
        data = []
        token_length_list = []
        char_count = []

        if bos_mode in ["add_default_bos", "replace_with_bos"]:
            bos_token = tokenizer.bos_token_id
        elif bos_mode in ["add_default_eos", "replace_with_eos"]:
            bos_token = tokenizer.eos_token_id
        elif bos_mode in ["add_newline_token", "replace_with_newline_token"]:
            bos_token = tokenizer.encode("\n")[0]
        bos_tensor = torch.tensor([bos_token], device=model.device).unsqueeze(0)

        if track_byte_wise_data:
            max_byte_size = max(len(text.encode("utf-8")) for text in texts)
            byte_wise_loss_sum = [0.0 for _ in range(max_byte_size)]
            byte_wise_counts = [0 for _ in range(max_byte_size)]

        for idx, sample in tqdm(enumerate(texts), total=len(texts), desc="Evaluating"):

            inputs = tokenizer(sample, return_tensors="pt", add_special_tokens=False)
            inputs = inputs.to(model.device)
            seq_length = inputs["input_ids"].shape[-1]
            assert seq_length > 1, f"Sequence {sample} is too short"

            neg_log_prob_temp = 0
            for begin in range(0, seq_length, chunk_size):
                input_chunk = inputs["input_ids"][:, begin : begin + chunk_size]
                if bos_mode in ["add_default_bos", "add_default_eos", "add_newline_token"]:
                    input_chunk = torch.cat([bos_tensor, input_chunk], dim=-1)
                if bos_mode in ["replace_with_bos", "replace_with_eos", "replace_with_newline_token"]:
                    input_chunk[0, 0] = bos_token
                logit = model.forward(input_ids=input_chunk).logits[0, :, :]
                loss = self.calculate_log_sum(logit, input_chunk.squeeze(0))
                loss_sum = loss.sum().item()
                neg_log_prob_temp += loss_sum

            byte_index = 0
            if track_byte_wise_data:  # track byte-wise data is not compatible with chunking, so here we will get the whole sequence's loss
                # token_strings = tokenizer.convert_ids_to_tokens(input_chunk.squeeze(0).tolist()[1:])
                per_token_bytes = get_decoded(tokenizer, sample)
                all_bytes = [byte for token in per_token_bytes for byte in token]
                assert all_bytes == list(sample.encode("utf-8")), "All bytes are not the same"
                assert len(per_token_bytes) == loss.shape[0], "Number of tokens and loss are not the same"
                for l, byte_values in zip(loss, per_token_bytes):
                    # byte_values = [tokenizer.byte_decoder[c] for c in token_str]
                    per_byte_loss = l.item() / len(byte_values)
                    for _ in range(len(byte_values)):
                        byte_wise_loss_sum[byte_index] += per_byte_loss
                        byte_wise_counts[byte_index] += 1
                        byte_index += 1

            token_length_list.append(seq_length)
            data.append(neg_log_prob_temp)
            char_count.append(len(sample))

        avg_byte_wise_loss = []
        if track_byte_wise_data:
            for s, c in zip(byte_wise_loss_sum, byte_wise_counts):
                if c > 0:
                    avg_byte_wise_loss.append(s / c)
                else:
                    break

        data_dict = {
            "neg_log_prob_sum": sum(data) / len(data),
            "avg tokens": sum(token_length_list) / len(token_length_list),
            "avg character count": sum(char_count) / len(char_count),
            "parameters count": self.count_model_parameters_in_billions(model),
            "avg bytes": sum([self.get_string_byte_size(text) for text in texts]) / len(texts),
            "sample_count": len(texts),
        }

        if track_byte_wise_data:
            data_dict["byte_wise_loss"] = avg_byte_wise_loss
            data_dict["byte_wise_counts"] = byte_wise_counts
            factor = (1.0 / math.log(2.0)) * 0.125 * 100.0
            data_dict["byte_wise_compression_rate"] = [loss_val * factor for loss_val in avg_byte_wise_loss]

        return data_dict

    def make_log(self, data_dict, folder_path):
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                print(f"Directory created at {folder_path}")
            except Exception as e:
                print(f"Error creating directory: {e}")
                return

        model_name = data_dict["model_name_or_path"].split("/")[-1].replace(".pth", "")
        dataset_name = data_dict["data_path"].split("/")[-1]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{model_name}-{dataset_name}-{timestamp}.json"
        file_path = os.path.join(folder_path, file_name)

        try:
            with open(file_path, "w") as file:
                json.dump(data_dict, file, indent=4, default=self.default_serializer)
            print(f"Log saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving dictionary: {e}")

    @staticmethod
    def load_data_smart(dataset_name_or_path):

        if dataset_name_or_path.endswith(".jsonl"):
            with open(dataset_name_or_path, "r", encoding="utf-8") as file:
                data = [line for line in file]
            return [{"name": dataset_name_or_path, "texts": [json.loads(line)["content"] for line in data]}]
        elif dataset_name_or_path.endswith(".json"):
            with open(dataset_name_or_path, "r", encoding="utf-8") as file:
                return [{"name": dataset_name_or_path, "texts": json.load(file)}]
        else:
            from datasets import load_dataset
            from datasets.exceptions import DatasetNotFoundError

            try:
                from collections import defaultdict

                dataset = load_dataset(dataset_name_or_path)["test"]
                category_map = defaultdict(list)
                for row in dataset:
                    cat = f'{dataset_name_or_path}-{row["category"]}'
                    content = row["content"]
                    category_map[cat].append(content)
                return [{"name": category, "texts": texts} for category, texts in category_map.items()]
            except DatasetNotFoundError:
                print(f"Error: '{dataset_name_or_path}' is neither a local path nor a valid Hugging Face dataset name.")
                return []
            except FileNotFoundError:
                print(f"Error: '{dataset_name_or_path}' is neither a local path nor a valid Hugging Face dataset name.")
                return []
            except Exception as e:
                print(f"An error occurred while attempting to load dataset from Hugging Face: {e}")
                return []

    def evaluate(self, config: EvaluationConfig):

        # install requirements
        print(f"Installing requirements: {config.requirements}")
        if len(config.requirements) > 0:
            self.install_requirements(config.requirements)

        # load model
        print(f"Loading model {config.model_name_or_path}")
        if config.model_type == "hf":
            model, tokenizer = self.load_hf_model(config)
        elif config.model_type == "rwkv":
            model, tokenizer = self.load_rwkv(config)
        elif config.model_type == "rwkv7":
            model, tokenizer = self.load_rwkv7(config)
        elif config.model_type == "mamba":
            model, tokenizer = self.load_mamba(config)
        else:
            raise NotImplementedError

        for dataset_name_or_path in config.data:

            dataset = self.load_data_smart(dataset_name_or_path)

            for data in dataset:

                data_name = data["name"]
                texts = data["texts"]

                print("-" * 80)
                print(f"Evaluating {config.model_name_or_path} on {data_name}")

                # eval
                if config.model_type in ["hf", "mamba"]:
                    if config.batch_size > 1:
                        raise NotImplementedError
                    else:
                        results = self.eval_hf_model(
                            model=model,
                            tokenizer=tokenizer,
                            texts=texts,
                            chunk_size=config.chunk_size,
                            bos_mode=config.bos_mode,
                            track_byte_wise_data=config.track_byte_wise_data,
                        )
                elif config.model_type in ["rwkv", "rwkv7"]:
                    results = self.eval_rwkv(
                        model=model,
                        tokenizer=tokenizer,
                        texts=texts,
                        chunk_size=config.chunk_size,
                        bos_mode=config.bos_mode,
                        track_byte_wise_data=config.track_byte_wise_data,
                    )
                else:
                    raise NotImplementedError

                results["model_name_or_path"] = config.model_name_or_path
                results["tokenizer_name"] = config.tokenizer_name
                results["data_path"] = data_name
                results["chunk_size"] = config.chunk_size
                results["bos_mode"] = config.bos_mode
                results["model_args"] = config.model_args
                results["tokenizer_args"] = config.tokenizer_args
                results["requirements"] = config.requirements
                results["batch_size"] = config.batch_size
                results["enable_chunking"] = config.enable_chunking
                results["bpc"] = (results["neg_log_prob_sum"] / results["avg character count"]) * (1 / math.log(2))
                results["bpb"] = (results["neg_log_prob_sum"] / results["avg bytes"]) * (1 / math.log(2))
                results["compression_rate"] = results["neg_log_prob_sum"] / results["avg bytes"] * (1 / math.log(2)) * 0.125 * 100
                results["enable_chunking"] = config.enable_chunking
                results["track_byte_wise_data"] = config.track_byte_wise_data

                self.make_log(results, config.log_path)

                print(f"Finished evaluating {config.model_name_or_path} on {data_name}")
                skip_keys = {"byte_wise_loss", "byte_wise_counts", "byte_wise_compression_rate"}
                filtered_results = {k: v for k, v in results.items() if k not in skip_keys}
                print(json.dumps(filtered_results, indent=4, ensure_ascii=False, default=self.default_serializer))
                print("-" * 80)

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
