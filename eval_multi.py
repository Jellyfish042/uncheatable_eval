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
    EvaluationConfig(
        model_name_or_path='BlinkDL/rwkv-6-world/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
        tokenizer_name='rwkv_vocab_v20230424',
        model_type='rwkv',
        data=data_list
    ),
    EvaluationConfig(
        model_name_or_path='stabilityai/stablelm-2-1_6b',
        tokenizer_name='stabilityai/stablelm-2-1_6b',
        model_type='hf',
        data=data_list,
    ),
    # EvaluationConfig(
    #     model_name_or_path='state-spaces/mamba-2.8b-slimpj',
    #     tokenizer_name='EleutherAI/gpt-neox-20b',
    #     model_type='mamba',
    #     data=data_list,
    #     requirements=['mamba-ssm', 'causal-conv1d>=1.2.0']
    # ),
]

if __name__ == '__main__':
    for config in config_list:
        attempts = 3
        while attempts > 0:
            try:
                evaluator = Evaluator()
                evaluator.evaluate(config)
                break
            except Exception as e:
                print(f"Error: {e}")
                attempts -= 1
                if attempts > 0:
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print("All attempts failed.")
