from eval_multi import config_list
from helpers import install_requirements
import os

CACHE_DIR = './models/temp/'
os.environ['HF_AUTH_TOKEN'] = ''

os.environ['HF_HOME'] = CACHE_DIR

if __name__ == '__main__':
    for config in config_list:
        try:
            print(f'Downloading: {config.model_name_or_path}')
            if config.requirements:
                install_requirements(config.requirements)
            if config.model_type == 'hf':
                from transformers import AutoTokenizer, AutoModelForCausalLM

                hf_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name,
                                                             cache_dir=config.cache,
                                                             trust_remote_code=True)
                hf_model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path,
                                                                cache_dir=config.cache,
                                                                trust_remote_code=True)
            elif config.model_type == 'rwkv':
                from huggingface_hub import hf_hub_download

                repo_name = '/'.join(config.original_model_name_or_path.split('/', 2)[:-1])
                filename = config.original_model_name_or_path.split('/', 2)[-1]

                hf_hub_download(repo_id=repo_name,
                                filename=filename,
                                local_dir=config.cache,
                                cache_dir=config.cache,
                                trust_remote_code=True)
            elif config.model_type == 'mamba':
                from transformers import AutoTokenizer
                from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

                tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
                model = MambaLMHeadModel.from_pretrained(config.original_model_name_or_path)
            else:
                raise NotImplementedError
            print('Download successful')
            print('-' * 50)
        except Exception as e:
            print(f'Error: {e}')
            print('-' * 50)
