import subprocess
import os

model_names = [
    'BlinkDL/rwkv-6-world/RWKV-x060-World-1B6-v2-20240208-ctx4096.pth',
    'BlinkDL/rwkv-6-world/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
    'BlinkDL/rwkv-5-world/RWKV-5-World-1B5-v2-20231025-ctx4096.pth',
    'stabilityai/stablelm-2-1_6b',  # pip install accelerate tiktoken
    '42dot/42dot_LLM-PLM-1.3B',
    'h2oai/h2o-danube-1.8b-base',
    'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
    'EleutherAI/pythia-1.4b-v0',
    'bigscience/bloom-1b7',
    'BlinkDL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040.pth',
    'BlinkDL/rwkv-4-world/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth',
    '42dot/42dot_LLM-SFT-1.3B',
    'allenai/OLMo-1B',  # pip install ai2-olmo
    'Qwen/Qwen1.5-1.8B',
    'Qwen/Qwen-1_8B',  # pip install transformers_stream_generator einops
    'mosaicml/mpt-1b-redpajama-200b',
    'princeton-nlp/Sheared-LLaMA-1.3B',
    'tiiuae/falcon-rw-1b',
    'bigscience/bloomz-1b7',
    'OpenNLPLab/TransNormerLLM-1B',  # pip install sentencepiece
    'microsoft/phi-1_5',
    'state-spaces/mamba-1.4b-hf',
]

cache_dir = '../models/temp/'

for name in model_names:
    args = [
        'python', 'download_single_model.py',
        '--model_name', name,
        '--cache_dir', cache_dir,
    ]

    proc = subprocess.Popen(args, stdin=subprocess.PIPE, text=True)

    proc.communicate(input='y\n')