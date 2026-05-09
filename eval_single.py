from evaluator import EvaluationConfig, Evaluator
import torch

# Create an EvaluationConfig instance to evaluate your model, for example:
config = (
    EvaluationConfig(
        model_name_or_path="Qwen/Qwen3-0.6B-Base",
        tokenizer_name="Qwen/Qwen3-0.6B-Base",
        model_type="hf",
        data=["Jellyfish042/UncheatableEval-2026-04"],
        bos_mode="add_default_eos",
        model_args={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
    ),
)

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate(config)
