from evaluator import EvaluationConfig, Evaluator
import time
import torch
import gc

data_list = [
    "Jellyfish042/UncheatableEval-2025-12",
]

config_list = [
    EvaluationConfig(
        model_name_or_path="tiiuae/Falcon-H1-7B-Base",
        tokenizer_name="tiiuae/Falcon-H1-7B-Base",
        model_type="hf",
        data=data_list,
        model_args={"torch_dtype": torch.bfloat16},
        enable_chunking=False,
        track_byte_wise_data=True,
    ),
]

if __name__ == "__main__":
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
