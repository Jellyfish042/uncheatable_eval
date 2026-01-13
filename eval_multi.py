from evaluator import EvaluationConfig, Evaluator
import time
import torch
import gc

data_list = [
    # "Jellyfish042/UncheatableEval-2025-12-Long",
    # "Jellyfish042/UncheatableEval-2025-12",
    "support/uncheatable_eval_dev.jsonl",
]

MY_CACHE_DIR = None
RWKV_CACHE_DIR = None

config_list = [
    EvaluationConfig(
        model_name_or_path="rwkv7-g1c-1.5b-20260110-ctx8192.pth",
        tokenizer_name="rwkv_vocab_v20230424",
        model_type="rwkv7",
        data=data_list,
        enable_chunking=False,
        track_byte_wise_data=True,
        track_single_sample_byte_wise_data=True,
    ),
    EvaluationConfig(
        model_name_or_path="Qwen/Qwen3-1.7B-Base",
        tokenizer_name="Qwen/Qwen3-1.7B-Base",
        model_type="hf",
        data=data_list,
        cache=MY_CACHE_DIR,
        bos_mode="add_newline_token",
        model_args={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        enable_chunking=False,
        track_byte_wise_data=True,
        track_single_sample_byte_wise_data=True,
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
                import traceback

                print(f"Error occurred: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                traceback.print_exc()
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
