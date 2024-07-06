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


@dataclass
class EvaluationConfig:
    model_name_or_path: str
    tokenizer_name: str
    model_type: str
    data: list[str]

    model_args: Dict[Any, Any] = field(
        default_factory=dict)  # other arguments that can be passed to the Hugging Face AutoModelForCausalLM
    tokenizer_args: Dict[Any, Any] = field(
        default_factory=dict)  # other arguments that can be passed to the Hugging Face AutoTokenizer

    requirements: list[str] = field(default_factory=list)  # list of packages, will be installed automatically

    add_bos: bool = False  # whether to add bos token to the input sequence
    log_path: str = './logs/'  # path to save the evaluation results
    cache: str = './models/temp/'  # cache directory for the models
    chunk_size: int = 1024  # input tokens will be split into chunks of this size

    def __post_init__(self):
        default_model_args = {'device_map': 'auto', 'trust_remote_code': True}
        self.model_args = {**default_model_args, **self.model_args}

        default_tokenizer_args = {'trust_remote_code': True}
        self.tokenizer_args = {**default_tokenizer_args, **self.tokenizer_args}

        if not os.path.exists(self.model_name_or_path):
            if '.pth' in self.model_name_or_path and 'rwkv' in self.model_name_or_path.lower() and self.cache:
                self.model_name_or_path = os.path.join(self.cache, self.model_name_or_path.split('/')[-1])


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
    def default_serializer(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

    @staticmethod
    def install_requirements(requirements: list[str]) -> None:
        """
        Installs or upgrades packages based on the version requirements specified in the list.

        :param requirements: List of packages with potential version specifiers.
        """
        for requirement in requirements:
            package_info = requirement.split('==') if '==' in requirement else (
                requirement.split('>=') if '>=' in requirement else (
                    requirement.split('<=') if '<=' in requirement else [requirement]))
            package_name = package_info[0]
            required_version_spec = requirement[len(package_name):]

            try:
                # Check if the package is already installed
                dist = importlib.metadata.distribution(package_name)
                installed_version = version.parse(dist.version)
                if required_version_spec:
                    # Extract the operator and the version from requirement
                    operator = required_version_spec[:2] if required_version_spec[1] in ['=', '>'] else \
                        required_version_spec[0]
                    required_version = version.parse(required_version_spec.lstrip(operator))

                    # Version comparison based on the operator
                    if ((operator == '==' and installed_version == required_version) or
                            (operator == '>=' and installed_version >= required_version) or
                            (operator == '<=' and installed_version <= required_version)):
                        print(f"Package {package_name} already installed and meets the requirement {requirement}.")
                    else:
                        print(
                            f"Package {package_name} version {installed_version} does not meet the requirement {requirement}, upgrading...")
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", f"{package_name}{required_version_spec}"])
                else:
                    print(f"Package {package_name} is already installed.")
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
        hf_model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path,
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

        rwkv_model = RWKV(model=config.model_name_or_path, strategy='cuda fp16')
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
        model = MambaLMHeadModel.from_pretrained(config.model_name_or_path, device="cuda", dtype=torch.float16)
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

        if add_bos:
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
                    for begin in range(0, seq_length, chunk_size - len_bos):
                        input_chunk = inputs['input_ids'][:, begin: begin + chunk_size - len_bos]

                        input_chunk = torch.cat([torch.tensor([bos_token], device=input_chunk.device), input_chunk],
                                                dim=-1)

                        logit = model.forward(input_ids=input_chunk).logits[0, :, :]
                        # print(logit.shape, input_chunk.squeeze(0).shape)
                        # print(logit[len_bos:, :].shape, input_chunk.squeeze(0)[len_bos:].shape)

                        log_sum = self.calculate_log_sum(logit[len_bos:, :],
                                                         input_chunk.squeeze(0)[len_bos:])  # exclude bos
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

    # @torch.no_grad()
    # def eval_hf_model(self, model, tokenizer, texts, chunk_size, add_bos, batch_size=4):
    #     data = []
    #     token_length_list = []
    #     char_count = []
    #     all_input_chunks = []
    #
    #     if add_bos:
    #         bos_token = tokenizer.encode(tokenizer.bos_token)
    #         len_bos = len(bos_token)
    #
    #     if tokenizer.pad_token_id is not None:
    #         pad_token_id = tokenizer.pad_token_id
    #     elif tokenizer.eos_token_id is not None:
    #         pad_token_id = tokenizer.eos_token_id
    #     else:
    #         raise ValueError("Tokenizer does not have a pad_token_id or eos_token_id")
    #
    #     for idx, sample in tqdm(enumerate(texts), total=len(texts), desc='Tokenizing'):
    #
    #         char_count.append(len(sample))
    #
    #         inputs = tokenizer(sample, return_tensors='pt')
    #         inputs = inputs.to(model.device)
    #
    #         seq_length = inputs['input_ids'].shape[-1]
    #
    #         neg_log_prob_temp = 0
    #         if add_bos:
    #             for begin in range(0, seq_length, chunk_size - len_bos):
    #                 input_chunk = inputs['input_ids'][:, begin: begin + chunk_size - len_bos]
    #
    #                 token_length_list.append(input_chunk.shape[-1])
    #
    #                 input_chunk = torch.cat([torch.tensor([bos_token], device=input_chunk.device), input_chunk],
    #                                         dim=-1)
    #
    #                 pad_size = chunk_size - input_chunk.shape[-1]
    #                 if pad_size > 0:
    #                     padded_chunk = F.pad(input_chunk, (0, pad_size), "constant", pad_token_id)
    #                     all_input_chunks.append(padded_chunk)
    #                 else:
    #                     all_input_chunks.append(input_chunk)
    #
    #                 # print(logit.shape, input_chunk.squeeze(0).shape)
    #                 # print(logit[len_bos:, :].shape, input_chunk.squeeze(0)[len_bos:].shape)
    #
    #         else:
    #             for begin in range(0, seq_length, chunk_size):
    #                 input_chunk = inputs['input_ids'][:, begin: begin + chunk_size]
    #
    #                 token_length_list.append(input_chunk.shape[-1])
    #
    #                 pad_size = chunk_size - input_chunk.shape[-1]
    #                 if pad_size > 0:
    #                     padded_chunk = F.pad(input_chunk, (pad_size, 0), "constant", pad_token_id)
    #                     all_input_chunks.append(padded_chunk)
    #                 else:
    #                     all_input_chunks.append(input_chunk)
    #
    #     all_input_chunks_tensor = torch.cat(all_input_chunks, dim=0)
    #
    #     # Process the input chunks in batches
    #     for i in tqdm(range(0, len(all_input_chunks_tensor), batch_size),
    #                   total=math.ceil(len(all_input_chunks_tensor) / batch_size),
    #                   desc='Inference'):
    #         batch = all_input_chunks_tensor[i:i + batch_size]
    #         attention_mask = batch != pad_token_id
    #         outputs = model(input_ids=batch, attention_mask=attention_mask)
    #         logits = outputs.logits
    #
    #         # Calculate the negative log probability for each chunk
    #         for j in range(batch.shape[0]):
    #             input_ids = batch[j]
    #             logit = logits[j]
    #
    #             mask = input_ids != pad_token_id
    #
    #             masked_logits = logit[mask]
    #             masked_input_ids = input_ids[mask]
    #
    #             neg_log_prob = self.calculate_log_sum(masked_logits, masked_input_ids)
    #             data.append(neg_log_prob)
    #
    #     data_dict = {
    #         'neg_log_prob_sum': sum(data) / len(texts),
    #         'avg tokens': sum(token_length_list) / len(texts),
    #         'avg character count': sum(char_count) / len(texts),
    #         'parameters count': self.count_model_parameters_in_billions(model),
    #         'avg bytes': sum([self.get_string_byte_size(text) for text in texts]) / len(texts),
    #         'sample_count': len(texts)
    #     }
    #
    #     # print(f'log probability sum: {sum(data) / len(data):.2f}')
    #     # print(f'avg tokens: {sum(token_length_list) / len(token_length_list):.0f}')
    #
    #     return data_dict

    def make_log(self, data_dict, folder_path):
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
                json.dump(data_dict, file, indent=4, default=self.default_serializer)
            print(f"Dictionary saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving dictionary: {e}")

    def evaluate(self, config: EvaluationConfig):

        # install requirements
        if len(config.requirements) > 0:
            self.install_requirements(config.requirements)

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

            print(f'Evaluating {config.model_name_or_path} on {data_file}')

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

            results['model_name_or_path'] = config.model_name_or_path
            results['tokenizer_name'] = config.tokenizer_name
            results['data_path'] = data_file
            results['chunk_size'] = config.chunk_size
            results['add_bos'] = config.add_bos
            results['model_args'] = config.model_args
            results['tokenizer_args'] = config.tokenizer_args
            results['requirements'] = config.requirements

            self.make_log(results, config.log_path)

            print(f'Finished evaluating {config.model_name_or_path} on {data_file}')
            print(json.dumps(results, indent=4, ensure_ascii=False, default=self.default_serializer))

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
