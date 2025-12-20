from evaluator import EvaluationConfig, Evaluator

# Create an EvaluationConfig instance to evaluate your model, for example:
config = EvaluationConfig(
    model_name_or_path="stabilityai/stablelm-2-1_6b",
    tokenizer_name="stabilityai/stablelm-2-1_6b",
    model_type="hf",
    data=["Jellyfish042/UncheatableEval-2025-12"],
)

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate(config)
