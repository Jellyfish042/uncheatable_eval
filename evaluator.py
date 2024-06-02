import importlib.metadata
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any
import subprocess
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class EvaluationConfig:
    model_name: str
    tokenizer_name: str
    model_type: str
    data: list[str]

    model_args: Dict[Any, Any] = field(default_factory=lambda: {
        'device_map': 'auto',
        'trust_remote_code': True,
    })
    tokenizer_args: Dict[str, str] = field(default_factory=dict)

    requirements: list[str] = field(default_factory=list)

    add_bos: bool = False
    log_path: str = './logs/'
    cache: str = './models/temp/'
    chunk_size: int = 1024


class Evaluator:
    def __init__(self):
        pass

    @staticmethod
    def load_list_from_json(file_path):
        """
        Loads a list of strings from a JSON file.

        :param file_path: Path of the JSON file to be loaded.
        :return: List of strings loaded from the JSON file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    @staticmethod
    def install_requirements(requirements: list[str]) -> None:
        """
        Installs all packages listed in requirements.

        :param requirements: List of packages to be installed.
        """
        for package in requirements:
            package_name = package.split('==')[0]
            required_version = package.split('==')[1]
            try:
                # Check if the package is already installed
                dist = importlib.metadata.distribution(package_name)
                installed_version = dist.version
                if '==' in package:
                    if installed_version == required_version:
                        print(f"Package {package} is already installed and meets the requirement.")
                    else:
                        raise RuntimeError(f"Package {package_name} version conflict: {installed_version} != {required_version}")
                else:
                    print(f"Package {package} is already installed and meets the requirement.")
            except importlib.metadata.PackageNotFoundError:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Error installing package {package}: {e}")

    @staticmethod
    def print_model_parameters_in_billions(model):
        total_params = sum(p.numel() for p in model.parameters())

        total_params_billion = total_params / 1e9

        print(f"Model parameters: {total_params_billion:.3f} billion")

    @staticmethod
    def print_rwkv_parameters_in_billions(rwkv_model):
        total_params = 0
        for param in rwkv_model.w.values():
            total_params += param.numel()
        print(f"Model parameters: {total_params / 1e9:.3f} billion")

    def load_hf_model(self, config: EvaluationConfig):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        hf_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name,
                                                     cache_dir=config.cache,
                                                     **config.tokenizer_args)
        hf_model = AutoModelForCausalLM.from_pretrained(config.model_name,
                                                        cache_dir=config.cache,
                                                        **config.model_args
                                                        ).eval()

        self.print_model_parameters_in_billions(hf_model)

        return hf_model, hf_tokenizer

    def load_rwkv(self, config: EvaluationConfig):
        os.environ['RWKV_JIT_ON'] = '1'
        os.environ["RWKV_CUDA_ON"] = '1'

        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE

        rwkv_model = RWKV(model=config.model_name, strategy='cuda fp16')
        rwkv_pipeline = PIPELINE(rwkv_model, config.tokenizer_name)
        rwkv_tokenizer = rwkv_pipeline.tokenizer

        self.print_rwkv_parameters_in_billions(rwkv_model)

        return rwkv_model, rwkv_tokenizer

    def load_mamba(self, config: EvaluationConfig):
        # state-spaces/mamba-2.8b-slimpj
        # pip install mamba-ssm
        # pip install causal-conv1d>=1.2.0
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        model = MambaLMHeadModel.from_pretrained(config.model_name, device="cuda", dtype=torch.float16)
        model.device = torch.device('cuda')

        self.print_model_parameters_in_billions(model)

        return model, tokenizer

    @staticmethod
    def calculate_log_sum(logits, target_token_ids):
        shifted_logits = logits[:-1, :]
        shifted_targets = target_token_ids[1:]

        log_probs = F.log_softmax(shifted_logits, dim=-1)

        target_log_probs = -log_probs.gather(1, shifted_targets.unsqueeze(1)).squeeze()
        # print(target_log_probs)

        log_sum = torch.sum(target_log_probs, dim=-1)
        # print(perplexity_sum)

        return log_sum.item()

    @staticmethod
    def count_rwkv_parameters_in_billions(rwkv_model):
        total_params = 0
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
        byte_array = input_string.encode('utf-8')
        # Calculate the length of the byte array
        byte_size = len(byte_array)
        return byte_size

    def eval_rwkv(self, model, tokenizer, texts, chunk_size):
        rwkv_test_data = []
        rwkv_token_length_list = []
        char_count = []

        for idx, sample in tqdm(enumerate(texts), total=len(texts)):

            char_count.append(len(sample))

            with torch.no_grad():

                tokenized = tokenizer.encode(sample)
                if hasattr(tokenized, 'ids'):
                    input_seq = tokenized.ids  # RWKV v4pile
                else:
                    input_seq = tokenized  # RWKV world

                input_length = len(input_seq)

                neg_log_prob_temp = 0
                for begin in range(0, input_length, chunk_size):
                    input_chunk = input_seq[begin: begin + chunk_size]

                    logit = model.forward(input_chunk, None, full_output=True)[0]

                    if len(input_chunk) == 1:
                        logit = logit.unsqueeze(0)

                    log_sum = self.calculate_log_sum(logit, torch.tensor(input_chunk).cuda())

                    neg_log_prob_temp += log_sum

                rwkv_token_length_list.append(input_length)
                rwkv_test_data.append(neg_log_prob_temp)

        data_dict = {
            'neg_log_prob_sum': sum(rwkv_test_data) / len(rwkv_test_data),
            'avg tokens': sum(rwkv_token_length_list) / len(rwkv_token_length_list),
            'avg character count': sum(char_count) / len(char_count),
            'parameters count': self.count_rwkv_parameters_in_billions(model),
            'avg bytes': sum([self.get_string_byte_size(text) for text in texts]) / len(texts),
            'sample_count': len(texts)
        }

        # print(f'log probability sum: {sum(rwkv_test_data) / len(rwkv_test_data):.2f}')
        # print(f'avg tokens: {sum(rwkv_token_length_list) / len(rwkv_token_length_list):.0f}')

        return data_dict

    def eval_hf_model(self, model, tokenizer, texts, chunk_size, add_bos):
        data = []
        token_length_list = []
        char_count = []

        bos_token = tokenizer.encode(tokenizer.bos_token)
        len_bos = len(bos_token)

        for idx, sample in tqdm(enumerate(texts), total=len(texts)):

            char_count.append(len(sample))

            with torch.no_grad():

                inputs = tokenizer(sample, return_tensors='pt')
                inputs = inputs.to(model.device)

                seq_length = inputs['input_ids'].shape[-1]

                neg_log_prob_temp = 0
                if add_bos:
                    for begin in range(0, seq_length - len_bos, chunk_size):
                        input_chunk = inputs['input_ids'][:, begin: begin + chunk_size - len_bos]

                        input_chunk = torch.cat([torch.tensor([bos_token], device=input_chunk.device), input_chunk],
                                                dim=-1)

                        logit = model.forward(input_ids=input_chunk).logits[0, :, :]
                        # print(logit.shape, input_chunk.squeeze(0).shape)
                        # print(logit[len_bos:, :].shape, input_chunk.squeeze(0)[len_bos:].shape)

                        log_sum = self.calculate_log_sum(logit[len_bos:, :], input_chunk.squeeze(0)[len_bos:])  # exclude bos
                        neg_log_prob_temp += log_sum
                else:
                    for begin in range(0, seq_length, chunk_size):
                        input_chunk = inputs['input_ids'][:, begin: begin + chunk_size]

                        logit = model.forward(input_ids=input_chunk).logits[0, :, :]

                        log_sum = self.calculate_log_sum(logit, input_chunk.squeeze(0))
                        neg_log_prob_temp += log_sum

                # neg_log_prob_temp = 0
                # for begin in range(0, seq_length, chunk_size):
                #     input_chunk = inputs['input_ids'][:, begin: begin + chunk_size]
                #
                #     logit = model.forward(input_ids=input_chunk).logits[0, :, :]
                #
                #     log_sum = self.calculate_log_sum(logit, input_chunk.squeeze(0))
                #     neg_log_prob_temp += log_sum

                token_length_list.append(seq_length)
                data.append(neg_log_prob_temp)

        data_dict = {
            'neg_log_prob_sum': sum(data) / len(data),
            'avg tokens': sum(token_length_list) / len(token_length_list),
            'avg character count': sum(char_count) / len(char_count),
            'parameters count': self.count_model_parameters_in_billions(model),
            'avg bytes': sum([self.get_string_byte_size(text) for text in texts]) / len(texts),
            'sample_count': len(texts)
        }

        # print(f'log probability sum: {sum(data) / len(data):.2f}')
        # print(f'avg tokens: {sum(token_length_list) / len(token_length_list):.0f}')

        return data_dict

    @staticmethod
    def make_log(data_dict, folder_path):
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                print(f"Directory created at {folder_path}")
            except Exception as e:
                print(f"Error creating directory: {e}")
                return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{timestamp}.json"
        file_path = os.path.join(folder_path, file_name)

        try:
            with open(file_path, 'w') as file:
                json.dump(data_dict, file, indent=4)
            print(f"Dictionary saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving dictionary: {e}")

    def evaluate(self, config: EvaluationConfig):

        # load model
        if config.model_type == 'hf':
            model, tokenizer = self.load_hf_model(config)
        elif config.model_type == 'rwkv':
            model, tokenizer = self.load_rwkv(config)
        elif config.model_type == 'mamba':
            model, tokenizer = self.load_mamba(config)
        else:
            raise NotImplementedError

        for data_file in config.data:

            print(f'Evaluating {config.model_name} on {data_file}')

            # load data
            texts = self.load_list_from_json(data_file)
            print(f'data size: {len(texts)}')

            # eval
            if config.model_type in ['hf', 'mamba']:
                results = self.eval_hf_model(model=model,
                                             tokenizer=tokenizer,
                                             texts=texts,
                                             chunk_size=config.chunk_size,
                                             add_bos=config.add_bos
                                             )
            elif config.model_type == 'rwkv':
                results = self.eval_rwkv(model=model, tokenizer=tokenizer, texts=texts, chunk_size=config.chunk_size)
            else:
                raise NotImplementedError

            results['model_name_or_path'] = config.model_name
            results['tokenizer_name'] = config.tokenizer_name
            results['data_path'] = data_file
            results['chunk_size'] = config.chunk_size
            results['add_bos'] = config.add_bos
            results['model_args'] = config.model_args
            results['tokenizer_args'] = config.tokenizer_args
            results['requirements'] = config.requirements

            self.make_log(results, config.log_path)

            print(f'Finished evaluating {config.model_name} on {data_file}')
            print(json.dumps(results, indent=4, ensure_ascii=False))
