from evaluator import EvaluationConfig, Evaluator
import time
import torch
import gc

data_list = [
    'data/ao3_english_20240601to20240615.json',
    'data/arxiv_computer_science_20240601to20240615.json',
    'data/arxiv_physics_20240601to20240615.json',
    'data/bbc_news_20240601to20240615.json',
    'data/github_cpp_20240601to20240615.json',
    'data/github_python_20240601to20240615.json',
    'data/wikipedia_english_20240601to20240615.json'
]

config_list = [
    # ~1.5B
    EvaluationConfig(
        model_name_or_path='Qwen/Qwen2-1.5B',
        tokenizer_name='Qwen/Qwen2-1.5B',
        model_type='hf',
        data=data_list,
    ),
    # EvaluationConfig(
    #     model_name_or_path='stabilityai/stablelm-2-1_6b',
    #     tokenizer_name='stabilityai/stablelm-2-1_6b',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-6-world/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
    #     tokenizer_name='rwkv_vocab_v20230424',
    #     model_type='rwkv',
    #     data=data_list,
    #     requirements=['rwkv']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-5-world/RWKV-5-World-1B5-v2-20231025-ctx4096.pth',
    #     tokenizer_name='rwkv_vocab_v20230424',
    #     model_type='rwkv',
    #     data=data_list,
    #     requirements=['rwkv']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='h2oai/h2o-danube-1.8b-base',
    #     tokenizer_name='h2oai/h2o-danube-1.8b-base',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
    #     tokenizer_name='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='EleutherAI/pythia-1.4b-v0',
    #     tokenizer_name='EleutherAI/pythia-1.4b-v0',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='bigscience/bloom-1b7',
    #     tokenizer_name='bigscience/bloom-1b7',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040.pth',
    #     tokenizer_name='./support/20B_tokenizer.json',
    #     model_type='rwkv',
    #     data=data_list,
    #     requirements=['rwkv']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-4-world/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth',
    #     tokenizer_name='rwkv_vocab_v20230424',
    #     model_type='rwkv',
    #     data=data_list,
    #     requirements=['rwkv']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='allenai/OLMo-1B-hf',
    #     tokenizer_name='allenai/OLMo-1B-hf',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='Qwen/Qwen1.5-1.8B',
    #     tokenizer_name='Qwen/Qwen1.5-1.8B',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='Qwen/Qwen-1_8B',
    #     tokenizer_name='Qwen/Qwen-1_8B',
    #     model_type='hf',
    #     data=data_list,
    #     requirements=['transformers_stream_generator'],
    # ),
    # EvaluationConfig(
    #     model_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B',
    #     tokenizer_name='princeton-nlp/Sheared-LLaMA-1.3B',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='tiiuae/falcon-rw-1b',
    #     tokenizer_name='tiiuae/falcon-rw-1b',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='OpenNLPLab/TransNormerLLM-1B',
    #     tokenizer_name='OpenNLPLab/TransNormerLLM-1B',
    #     model_type='hf',
    #     data=data_list,
    #     requirements=['sentencepiece'],
    # ),
    # EvaluationConfig(
    #     model_name_or_path='microsoft/phi-1_5',
    #     tokenizer_name='microsoft/phi-1_5',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='state-spaces/mamba-1.4b-hf',
    #     tokenizer_name='state-spaces/mamba-1.4b-hf',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='apple/OpenELM-1_1B',
    #     tokenizer_name='meta-llama/Llama-2-7b-hf',
    #     model_type='hf',
    #     data=data_list,
    #     add_bos=True,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='IndexTeam/Index-1.9B',
    #     tokenizer_name='IndexTeam/Index-1.9B',
    #     model_type='hf',
    #     data=data_list,
    # ),

    # ~3B
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-6-world/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
    #     tokenizer_name='rwkv_vocab_v20230424',
    #     model_type='rwkv',
    #     data=data_list,
    #     requirements=['rwkv']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-5-world/RWKV-5-World-3B-v2-20231113-ctx4096.pth',
    #     tokenizer_name='rwkv_vocab_v20230424',
    #     model_type='rwkv',
    #     data=data_list,
    #     requirements=['rwkv']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='stabilityai/stablelm-3b-4e1t',
    #     tokenizer_name='stabilityai/stablelm-3b-4e1t',
    #     model_type='hf',
    #     data=data_list
    # ),
    # EvaluationConfig(
    #     model_name_or_path='Qwen/Qwen1.5-4B',
    #     tokenizer_name='Qwen/Qwen1.5-4B',
    #     model_type='hf',
    #     data=data_list
    # ),
    # EvaluationConfig(
    #     model_name_or_path='state-spaces/mamba-2.8b-hf',
    #     tokenizer_name='state-spaces/mamba-2.8b-hf',
    #     model_type='hf',
    #     data=data_list
    # ),
    # EvaluationConfig(
    #     model_name_or_path='openlm-research/open_llama_3b_v2',
    #     tokenizer_name='openlm-research/open_llama_3b_v2',
    #     model_type='hf',
    #     data=data_list
    # ),
    # EvaluationConfig(
    #     model_name_or_path='cerebras/btlm-3b-8k-base',
    #     tokenizer_name='cerebras/btlm-3b-8k-base',
    #     model_type='hf',
    #     data=data_list
    # ),
    # EvaluationConfig(
    #     model_name_or_path='state-spaces/mamba-2.8b-slimpj',
    #     tokenizer_name='EleutherAI/gpt-neox-20b',
    #     model_type='mamba',
    #     data=data_list,
    #     requirements=['mamba-ssm', 'causal-conv1d>=1.2.0']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='EleutherAI/pythia-2.8b-v0',
    #     tokenizer_name='EleutherAI/pythia-2.8b-v0',
    #     model_type='hf',
    #     data=data_list
    # ),
    # EvaluationConfig(
    #     model_name_or_path='togethercomputer/RedPajama-INCITE-Base-3B-v1',
    #     tokenizer_name='togethercomputer/RedPajama-INCITE-Base-3B-v1',
    #     model_type='hf',
    #     data=data_list
    # ),
    # EvaluationConfig(
    #     model_name_or_path='princeton-nlp/Sheared-LLaMA-2.7B',
    #     tokenizer_name='princeton-nlp/Sheared-LLaMA-2.7B',
    #     model_type='hf',
    #     data=data_list
    # ),
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096.pth',
    #     tokenizer_name='./support/20B_tokenizer.json',
    #     model_type='rwkv',
    #     data=data_list,
    #     requirements=['rwkv']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-4-world/RWKV-4-World-3B-v1-20230619-ctx4096.pth',
    #     tokenizer_name='rwkv_vocab_v20230424',
    #     model_type='rwkv',
    #     data=data_list,
    #     requirements=['rwkv']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='apple/OpenELM-3B',
    #     tokenizer_name='meta-llama/Llama-2-7b-hf',
    #     model_type='hf',
    #     data=data_list,
    #     add_bos=True,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='microsoft/Phi-3-mini-4k-instruct',
    #     tokenizer_name='microsoft/Phi-3-mini-4k-instruct',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='google/gemma-2b',
    #     tokenizer_name='google/gemma-2b',
    #     model_type='hf',
    #     data=data_list,
    #     add_bos=True,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='google/recurrentgemma-2b',
    #     tokenizer_name='google/recurrentgemma-2b',
    #     model_type='hf',
    #     data=data_list,
    #     add_bos=True,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='microsoft/phi-2',
    #     tokenizer_name='microsoft/phi-2',
    #     model_type='hf',
    #     data=data_list,
    # ),

    # ~7B
    # EvaluationConfig(
    #     model_name_or_path='meta-llama/Meta-Llama-3-8B',
    #     tokenizer_name='meta-llama/Meta-Llama-3-8B',
    #     model_type='hf',
    #     data=data_list,
    #     requirements=['accelerate']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='mistralai/Mistral-7B-v0.1',
    #     tokenizer_name='mistralai/Mistral-7B-v0.1',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='Qwen/Qwen2-7B',
    #     tokenizer_name='Qwen/Qwen2-7B',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='Qwen/Qwen1.5-7B',
    #     tokenizer_name='Qwen/Qwen1.5-7B',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-6-world/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
    #     tokenizer_name='BlinkDL/rwkv-6-world/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
    #     model_type='rwkv',
    #     data=data_list,
    #     requirements=['rwkv']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-5-world/RWKV-5-World-7B-v2-20240128-ctx4096.pth',
    #     tokenizer_name='BlinkDL/rwkv-5-world/RWKV-5-World-7B-v2-20240128-ctx4096.pth',
    #     model_type='rwkv',
    #     data=data_list,
    #     requirements=['rwkv']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='meta-llama/Llama-2-7b-hf',
    #     tokenizer_name='meta-llama/Llama-2-7b-hf',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='TRI-ML/mamba-7b-rw',
    #     tokenizer_name='TRI-ML/mamba-7b-rw',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='tiiuae/falcon-7b',
    #     tokenizer_name='tiiuae/falcon-7b',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='mosaicml/mpt-7b',
    #     tokenizer_name='mosaicml/mpt-7b',
    #     model_type='hf',
    #     data=data_list,
    #     requirements=['einops']
    # ),
    # EvaluationConfig(
    #     model_name_or_path='EleutherAI/pythia-6.9b-v0',
    #     tokenizer_name='EleutherAI/pythia-6.9b-v0',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='allenai/OLMo-1.7-7B-hf',
    #     tokenizer_name='allenai/OLMo-1.7-7B-hf',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='google/gemma-7b',
    #     tokenizer_name='google/gemma-7b',
    #     model_type='hf',
    #     data=data_list,
    #     add_bos=True,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='microsoft/Phi-3-small-8k-instruct',
    #     tokenizer_name='microsoft/Phi-3-small-8k-instruct',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='openlm-research/open_llama_7b_v2',
    #     tokenizer_name='openlm-research/open_llama_7b_v2',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='Zyphra/Zamba-7B-v1',
    #     tokenizer_name='Zyphra/Zamba-7B-v1',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='m-a-p/neo_7b',
    #     tokenizer_name='m-a-p/neo_7b',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='CohereForAI/aya-23-8B',
    #     tokenizer_name='CohereForAI/aya-23-8B',
    #     model_type='hf',
    #     data=data_list,
    # ),

    # ~14B
    # EvaluationConfig(
    #     model_name_or_path='Qwen/Qwen1.5-14B',
    #     tokenizer_name='Qwen/Qwen1.5-14B',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='meta-llama/Llama-2-13b-hf',
    #     tokenizer_name='meta-llama/Llama-2-13b-hf',
    #     model_type='hf',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='EleutherAI/pythia-12b-v0',
    #     tokenizer_name='EleutherAI/pythia-12b-v0',
    #     model_type='hf',
    #     data=data_list,
    # ),
]

if __name__ == '__main__':
    success_models = []
    failed_models = []
    for config in config_list:
        attempts = 3
        model_name = config.model_name_or_path
        while attempts > 0:
            try:
                evaluator = Evaluator()
                evaluator.evaluate(config)
                success_models.append(model_name)
                break
            except Exception as e:
                print(f"Error: {e}")
                attempts -= 1
                if attempts > 0:
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print("All attempts failed.")
                    failed_models.append(model_name)
            finally:
                del evaluator
                gc.collect()
                torch.cuda.empty_cache()

    print("\nSummary:")
    print("----------------------------")
    print("Successful Models:")
    for model in success_models:
        print(f"- {model}")
    print("\nFailed Models:")
    for model in failed_models:
        print(f"- {model}")
