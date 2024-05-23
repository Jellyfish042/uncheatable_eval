# import packages
import os
from tqdm import tqdm
import warnings
import json
import torch.nn.functional as F
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import argparse

RWKV4_TOKENIZER_FILE = "./support/20B_tokenizer.json"


def load_list_from_json(file_path):
    """
    Loads a list of strings from a JSON file.

    :param file_path: Path of the JSON file to be loaded.
    :return: List of strings loaded from the JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def calculate_log_sum(logits, target_token_ids):
    shifted_logits = logits[:-1, :]
    shifted_targets = target_token_ids[1:]

    log_probs = F.log_softmax(shifted_logits, dim=-1)

    target_log_probs = -log_probs.gather(1, shifted_targets.unsqueeze(1)).squeeze()
    # print(target_log_probs)

    log_sum = torch.sum(target_log_probs, dim=-1)
    # print(perplexity_sum)

    return log_sum.item()


def print_model_parameters_in_billions(model):
    total_params = sum(p.numel() for p in model.parameters())

    total_params_billion = total_params / 1e9

    print(f"Model parameters: {total_params_billion:.3f} billion")


def get_model_parameters_in_billions(model):
    total_params = sum(p.numel() for p in model.parameters())

    total_params_billion = total_params / 1e9

    return total_params_billion


def count_rwkv_parameters_in_billions(rwkv_model):
    total_params = 0
    for param in rwkv_model.w.values():
        total_params += param.numel()
    return total_params / 1e9


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


def load_rwkv(path):
    os.environ['RWKV_JIT_ON'] = '1'
    os.environ["RWKV_CUDA_ON"] = '1'

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    rwkv_model = RWKV(model=path, strategy='cuda fp16')
    rwkv_pipeline = PIPELINE(rwkv_model, r"rwkv_vocab_v20230424")
    rwkv_tokenizer = rwkv_pipeline.tokenizer

    return rwkv_model, rwkv_tokenizer


def load_rwkv4pile(path):
    os.environ['RWKV_JIT_ON'] = '1'
    os.environ["RWKV_CUDA_ON"] = '1'

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    rwkv_model = RWKV(model=path, strategy='cuda fp16')
    rwkv_pipeline = PIPELINE(rwkv_model, RWKV4_TOKENIZER_FILE)
    rwkv_tokenizer = rwkv_pipeline.tokenizer

    return rwkv_model, rwkv_tokenizer


def load_hf_model(path, cache_path):
    hf_tokenizer = AutoTokenizer.from_pretrained(path, cache_dir=cache_path)
    if cache_path is not None:
        hf_model = AutoModelForCausalLM.from_pretrained(path,
                                                        device_map="auto",
                                                        trust_remote_code=True,
                                                        cache_dir=cache_path).eval()
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(path,
                                                        device_map="auto",
                                                        trust_remote_code=True).eval()

    print_model_parameters_in_billions(hf_model)

    return hf_model, hf_tokenizer


def load_mamba(path):
    if 'hf' in path:

        # state-spaces/mamba-1.4b-hf
        from transformers import MambaForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = MambaForCausalLM.from_pretrained(path).cuda()

        print_model_parameters_in_billions(model)

        return model, tokenizer
    else:
        # state-spaces/mamba-2.8b-slimpj
        # pip install git+https://github.com/huggingface/transformers@main
        # pip install mamba-ssm
        # pip install causal-conv1d>=1.2.0
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = MambaLMHeadModel.from_pretrained(path, device="cuda", dtype=torch.float16)
        model.device = torch.device('cuda')

        print_model_parameters_in_billions(model)

        return model, tokenizer


def eval_rwkv(model, tokenizer, texts, chunk_size, v4pile=False):
    rwkv_test_data = []
    rwkv_token_length_list = []
    char_count = []

    for idx, sample in tqdm(enumerate(texts), total=len(texts)):

        char_count.append(len(sample))

        with torch.no_grad():

            if v4pile:
                input_seq = tokenizer.encode(sample).ids  # v4
            else:
                input_seq = tokenizer.encode(sample)

            input_length = len(input_seq)

            neg_log_prob_temp = 0
            for begin in range(0, input_length, chunk_size):
                input_chunk = input_seq[begin: begin + chunk_size]

                logit = model.forward(input_chunk, None, full_output=True)[0]

                if len(input_chunk) == 1:
                    logit = logit.unsqueeze(0)

                log_sum = calculate_log_sum(logit, torch.tensor(input_chunk).cuda())

                neg_log_prob_temp += log_sum

            rwkv_token_length_list.append(input_length)
            rwkv_test_data.append(neg_log_prob_temp)

    data_dict = {
        'neg_log_prob_sum': sum(rwkv_test_data) / len(rwkv_test_data),
        'avg tokens': sum(rwkv_token_length_list) / len(rwkv_token_length_list),
        'avg character count': sum(char_count) / len(char_count),
        'parameters count': count_rwkv_parameters_in_billions(model)
    }

    # print(f'log probability sum: {sum(rwkv_test_data) / len(rwkv_test_data):.2f}')
    # print(f'avg tokens: {sum(rwkv_token_length_list) / len(rwkv_token_length_list):.0f}')

    return data_dict


def eval_hf_model(model, tokenizer, texts, chunk_size):
    data = []
    token_length_list = []
    char_count = []

    for idx, sample in tqdm(enumerate(texts), total=len(texts)):

        char_count.append(len(sample))

        with torch.no_grad():

            inputs = tokenizer(sample, return_tensors='pt')
            inputs = inputs.to(model.device)

            seq_length = inputs['input_ids'].shape[-1]

            neg_log_prob_temp = 0
            for begin in range(0, seq_length, chunk_size):
                input_chunk = inputs['input_ids'][:, begin: begin + chunk_size]

                logit = model.forward(input_ids=input_chunk).logits[0, :, :]

                log_sum = calculate_log_sum(logit, input_chunk.squeeze(0))
                neg_log_prob_temp += log_sum

            token_length_list.append(seq_length)
            data.append(neg_log_prob_temp)

    data_dict = {
        'neg_log_prob_sum': sum(data) / len(data),
        'avg tokens': sum(token_length_list) / len(token_length_list),
        'avg character count': sum(char_count) / len(char_count),
        'parameters count': get_model_parameters_in_billions(model)
    }

    # print(f'log probability sum: {sum(data) / len(data):.2f}')
    # print(f'avg tokens: {sum(token_length_list) / len(token_length_list):.0f}')

    return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='model name or path')
    parser.add_argument('--model_type', choices=['hf', 'rwkv', 'mamba', 'rwkv4pile'], required=True, help='model type')
    parser.add_argument('--data', type=str, required=True, help='data path (json file)')
    parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
    parser.add_argument('--model_cache', type=str, help='hugging face model cache')
    parser.add_argument('--chunk_size', type=int, default=1024, help='chunk size')

    args = parser.parse_args()

    # load data
    texts = load_list_from_json(args.data)
    print(f'data size: {len(texts)}')

    # load model
    if args.model_type == 'hf':
        model, tokenizer = load_hf_model(args.model, args.model_cache)
    elif args.model_type == 'rwkv':
        model, tokenizer = load_rwkv(args.model)
    elif args.model_type == 'mamba':
        model, tokenizer = load_mamba(args.model)
    elif args.model_type == 'rwkv4pile':
        model, tokenizer = load_rwkv4pile(args.model)
    else:
        raise NotImplementedError

    # eval
    if args.model_type in ['hf', 'mamba']:
        results = eval_hf_model(model=model, tokenizer=tokenizer, texts=texts, chunk_size=args.chunk_size)
    elif args.model_type == 'rwkv':
        results = eval_rwkv(model=model, tokenizer=tokenizer, texts=texts, chunk_size=args.chunk_size)
    elif args.model_type == 'rwkv4pile':
        results = eval_rwkv(model=model, tokenizer=tokenizer, texts=texts, chunk_size=args.chunk_size, v4pile=True)
    else:
        raise NotImplementedError

    results['model_name_or_path'] = args.model
    results['data_path'] = args.data
    results['chunk_size'] = args.chunk_size

    make_log(results, args.log_path)

    print(json.dumps(results, indent=4, ensure_ascii=False))
