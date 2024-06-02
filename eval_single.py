from evaluator import EvaluationConfig, Evaluator

# Create an EvaluationConfig instance to evaluate your model, for example:
config = EvaluationConfig(
    model_name_or_path='stabilityai/stablelm-2-1_6b',
    tokenizer_name='stabilityai/stablelm-2-1_6b',
    model_type='hf',
    data=['data/ao3_english_20240501to20240515.json']
)

if __name__ == '__main__':
    try:
        evaluator = Evaluator()
        evaluator.evaluate(config)
    except Exception as e:
        print(f"Error: {e}")
