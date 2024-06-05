from evaluator import EvaluationConfig, Evaluator
import time

data_list = [
    'data/ao3_english_20240501to20240515.json',
    'data/arxiv_computer_science_20240501to20240515.json',
    'data/arxiv_physics_20240501to20240515.json',
    'data/bbc_news_20240501to20240515.json',
    'data/github_cpp_20240501to20240515.json',
    'data/github_python_20240501to20240515.json',
    'data/wikipedia_english_20240501to20240515.json'
]

config_list = [
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-6-world/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
    #     tokenizer_name='rwkv_vocab_v20230424',
    #     model_type='rwkv',
    #     data=data_list
    # ),
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-5-world/RWKV-5-World-1B5-v2-20231025-ctx4096.pth',
    #     tokenizer_name='rwkv_vocab_v20230424',
    #     model_type='rwkv',
    #     data=data_list
    # ),
    EvaluationConfig(
        model_name_or_path='stabilityai/stablelm-2-1_6b',
        tokenizer_name='stabilityai/stablelm-2-1_6b',
        model_type='hf',
        data=data_list,
    ),
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
    #     tokenizer_name='BlinkDL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040.pth',
    #     model_type='rwkv',
    #     data=data_list,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='BlinkDL/rwkv-4-world/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth',
    #     tokenizer_name='BlinkDL/rwkv-4-world/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth',
    #     model_type='rwkv',
    #     data=data_list,
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
    #     model_name_or_path='apple/OpenELM-3B',
    #     tokenizer_name='meta-llama/Llama-2-7b-hf',
    #     model_type='hf',
    #     data=data_list,
    #     add_bos=True,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='google/gemma-2b',
    #     tokenizer_name='google/gemma-2b',
    #     model_type='hf',
    #     data=data_list,
    #     add_bos=True,
    # ),
    # EvaluationConfig(
    #     model_name_or_path='google/gemma-7b',
    #     tokenizer_name='google/gemma-7b',
    #     model_type='hf',
    #     data=data_list,
    #     add_bos=True,
    # ),
    #
    #
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
    #
    # EvaluationConfig(
    #     model_name_or_path='RWKV/rwkv-6-world-1b6',
    #     tokenizer_name='RWKV/rwkv-6-world-1b6',
    #     model_type='hf',
    #     data=data_list,
    #
    #     tokenizer_args={'trust_remote_code': True}
    # ),
    # EvaluationConfig(
    #     model_name_or_path='RWKV/rwkv-6-world-3b-v2.1',
    #     tokenizer_name='RWKV/rwkv-6-world-3b-v2.1',
    #     model_type='hf',
    #     data=data_list,
    #
    #     tokenizer_args={'trust_remote_code': True}
    # ),
    # EvaluationConfig(
    #     model_name_or_path='RWKV/rwkv-6-world-7b',
    #     tokenizer_name='RWKV/rwkv-6-world-7b',
    #     model_type='hf',
    #     data=data_list,
    #
    #     tokenizer_args={'trust_remote_code': True}
    # ),
    # EvaluationConfig(
    #     model_name_or_path='state-spaces/mamba-2.8b-slimpj',
    #     tokenizer_name='EleutherAI/gpt-neox-20b',
    #     model_type='mamba',
    #     data=data_list,
    #     requirements=['mamba-ssm', 'causal-conv1d>=1.2.0']
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
                del evaluator
                if attempts > 0:
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print("All attempts failed.")
                    failed_models.append(model_name)

    print("\nSummary:")
    print("----------------------------")
    print("Successful Models:")
    for model in success_models:
        print(f"- {model}")
    print("\nFailed Models:")
    for model in failed_models:
        print(f"- {model}")
