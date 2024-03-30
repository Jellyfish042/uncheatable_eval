import subprocess
import os

os.environ['HF_HOME'] = './models/temp/'

data_list = ['data/arxiv_pdfs_cs_240322to240328.json',
             'data/arxiv_pdfs_phy_240322to240328.json',
             'data/bbc_news_240322to240329.json']
hf_cache = './models/temp/'

models = [
    'BlinkDL/rwkv-6-world/RWKV-x060-World-1B6-v2-20240208-ctx4096.pth',
    # 'BlinkDL/rwkv-6-world/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
    # 'BlinkDL/rwkv-5-world/RWKV-5-World-1B5-v2-20231025-ctx4096.pth',
    # 'stabilityai/stablelm-2-1_6b',
    # '42dot/42dot_LLM-PLM-1.3B',
    # 'h2oai/h2o-danube-1.8b-base',
    # 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
    # 'EleutherAI/pythia-1.4b-v0',
    # 'bigscience/bloom-1b7',
    # 'BlinkDL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040.pth',
    # 'BlinkDL/rwkv-4-world/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth',
    # '42dot/42dot_LLM-SFT-1.3B',
    # 'allenai/OLMo-1B',
    # 'Qwen/Qwen1.5-1.8B',
    # 'Qwen/Qwen-1_8B',
    # 'mosaicml/mpt-1b-redpajama-200b',
    # 'princeton-nlp/Sheared-LLaMA-1.3B',
    # 'tiiuae/falcon-rw-1b',
    # 'bigscience/bloomz-1b7',
    # 'OpenNLPLab/TransNormerLLM-1B',
    # 'microsoft/phi-1_5',
    # 'state-spaces/mamba-1.4b-hf',  # pip install causal-conv1d>=1.2.0 mamba-ssm
]

for model in models:
    if 'BlinkDL/rwkv-4-pile' in model:
        model_type = 'rwkv4pile'
    elif 'BlinkDL/rwkv' in model:
        model_type = 'rwkv'
    elif 'mamba' in model:
        model_type = 'mamba'
    else:
        model_type = 'hf'
    for data in data_list:
        if 'BlinkDL/rwkv' in model:
            command = f"echo 'y' | python eval_model.py --model {os.path.join(hf_cache, model.split('/')[-1])} --model_type {model_type} --data {data} --model_cache {hf_cache}"
        else:
            command = f"echo 'y' | python eval_model.py --model {model} --model_type {model_type} --data {data} --model_cache {hf_cache}"

        subprocess.run(command, shell=True, text=True)
