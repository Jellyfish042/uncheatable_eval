from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import torch
import gc
import traceback
import os

if __name__ == '__main__':

    os.environ['HF_HOME'] = './models/temp/'

    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./models/temp/')
    parser.add_argument('--retry', type=int, default=5)
    args = parser.parse_args()

    # print(args)

    path = args.model_name
    cache_dir = args.cache_dir

    for _ in range(5):
        print(f'# downloading: {path}')
        try:
            if 'BlinkDL' in path:

                repo_name = '/'.join(path.split('/', 2)[:-1])
                filename = path.split('/', 2)[-1]

                hf_hub_download(repo_id=repo_name,
                                filename=filename,
                                local_dir=cache_dir,
                                cache_dir=cache_dir)
                print('# download successful')

                break
            elif path.startswith("state-spaces/mamba-") and 'hf' not in path:
                from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
                tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
                model = MambaLMHeadModel.from_pretrained(args.model_name, device='cpu')
            else:
                tokenizer = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)
                model = AutoModelForCausalLM.from_pretrained(path,
                                                             device_map="cpu",
                                                             force_download=False,
                                                             trust_remote_code=True,
                                                             cache_dir=cache_dir).eval()

                del model, tokenizer

                gc.collect()
                torch.cuda.empty_cache()
                print('# download successful')
                break

        except Exception as e:
            print(f'# {path} failed')
            traceback.print_exc()
