from evaluator import EvaluationConfig, Evaluator

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
    #     model_name='BlinkDL/rwkv-6-world/RWKV-x060-World-1B6-v2.1-20240328-ctx4096',
    #     tokenizer_name='rwkv_vocab_v20230424',
    #     model_type='rwkv',
    #     data=data_list
    # ),
    EvaluationConfig(
        model_name='stabilityai/stablelm-2-1_6b',
        tokenizer_name='stabilityai/stablelm-2-1_6b',
        model_type='hf',
        data=data_list,

        add_bos=True,
    )
]

if __name__ == '__main__':
    for config in config_list:
        try:
            evaluator = Evaluator()
            evaluator.evaluate(config)
        except Exception as e:
            print(f"Error: {e}")
