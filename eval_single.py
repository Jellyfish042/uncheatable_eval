from evaluator import EvaluationConfig, Evaluator
import torch

# Create an EvaluationConfig instance to evaluate your model, for example:
config = (
    EvaluationConfig(
        model_name_or_path="Qwen/Qwen3.5-0.8B-Base",
        tokenizer_name="Qwen/Qwen3.5-0.8B-Base",
        model_type="hf",
        data=["Jellyfish042/UncheatableEval-2026-10"],
        bos_mode="add_newline_token",
        model_args={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16},
    ),
)

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate(config)
