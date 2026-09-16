# Conventional-benchmark validation

This directory evaluates whether Uncheatable Eval has convergent validity with
standard base-model benchmarks. Correlation alone does not prove that two
benchmarks measure the same capability: model size, training compute, data
quality, and family membership affect both. The analysis protocol below tests
whether Uncheatable Eval adds predictive information beyond model size.

## Fixed protocols

Use exact checkpoints and tokenizers from the July 2026 leaderboard records.
Do not apply chat templates or instruction prompts.

| Benchmark | Tool | Protocol | Primary metric | Direction |
|---|---|---|---|---|
| MMLU | lm-evaluation-harness | 5-shot, choice log-likelihood | accuracy | higher is better |
| LAMBADA OpenAI | lm-evaluation-harness | 0-shot, last-word likelihood | exact-match accuracy | higher is better |
| HumanEval / HumanEval+ | EvalPlus | 0-shot base prompt, greedy, one sample | pass@1 | higher is better |

LAMBADA perplexity is retained as a diagnostic, but is not a primary cross-model
metric because token-level perplexity is tokenizer-dependent and is very close
to the likelihood quantity already measured by Uncheatable Eval. LAMBADA exact
last-word accuracy provides the more useful external criterion.

The harness task definitions are pinned through `requirements.txt`. MMLU uses
the `cais/mmlu` test split and its dev examples; `lambada_openai` scores the
space-prefixed final word. EvalPlus reports both original HumanEval and its
stronger added tests from the same generations.

## Environment and manifest

Use a separate environment so benchmark dependencies cannot change the
Uncheatable evaluator environment:

```powershell
conda create -n uncheatable-bench python=3.11 -y
conda run -n uncheatable-bench pip install -r benchmark_validation/requirements.txt

conda run -n torch2 python benchmark_validation/build_manifest.py `
  E:/codebase/UncheatableEvalLeaderboard/data/2026-07 `
  --output benchmark_validation/model_manifest.jsonl

conda run -n uncheatable-bench python benchmark_validation/pin_hf_revisions.py `
  --manifest benchmark_validation/model_manifest.jsonl
```

The manifest builder rejects inconsistent checkpoint/tokenizer metadata across
categories. In the current release it identifies 72 HF causal-LM candidates,
15 RWKV checkpoints, and 3 special Ministral checkpoints. Revision pinning may
require `huggingface-cli login` for gated repositories. Run a small smoke test
first; `--allow-unpinned` is acceptable only before revision pinning:

```powershell
conda run -n uncheatable-bench python benchmark_validation/run_lm_eval.py `
  --models "^(SmolLM2-135M|Qwen2.5-0.5B)$" --limit 10 --write-out --dry-run

conda run -n uncheatable-bench python benchmark_validation/run_lm_eval.py `
  --models "^(SmolLM2-135M|Qwen2.5-0.5B)$" --limit 10 --write-out
```

Inspect the printed prompts and sample logs. Then remove `--limit` and
`--write-out` for valid full-dataset scores; remove `--models` to run all HF
models. Limited and full runs use separate output directories.
`run_lm_eval.py` invokes MMLU and LAMBADA separately so their few-shot settings
cannot leak into one another. It writes sample-level logs and uses a completion
marker for restartable runs.

The existing Uncheatable result JSON records do not contain an immutable HF
revision. Pinning now prevents further drift, but cannot prove that the pinned
weights equal the earlier evaluation-time `main`. For the strict paper result,
recover the original cache snapshot if it still exists, or rerun Uncheatable
Eval on the pinned revisions used here.

## HumanEval

Generate code on the GPU host. `--force-base-prompt` is mandatory because some
base checkpoints ship a tokenizer with a chat template.

```powershell
conda run -n uncheatable-bench python benchmark_validation/run_evalplus_codegen.py `
  --models "^(SmolLM2-135M|Qwen2.5-0.5B)$" --dry-run

conda run -n uncheatable-bench python benchmark_validation/run_evalplus_codegen.py `
  --models "^(SmolLM2-135M|Qwen2.5-0.5B)$"
```

Do not execute generated Python directly on the host. Mount the generated result
directory into the EvalPlus Docker image and score each JSONL file there:

```powershell
docker run --rm --pull=always `
  -v "${PWD}/benchmark_validation/results/evalplus:/app" `
  ganler/evalplus:latest `
  evalplus.evaluate --dataset humaneval --samples /app/humaneval/MODEL_hf_temp_0.0.jsonl
```

Store the container image digest with the result. EvalPlus outputs original
HumanEval and HumanEval+ pass@1 from the same deterministic completion.

## Prefix policy

The conventional benchmarks use standard harness prompt tokenization. The
manifest maps `add_default_bos` to lm-eval's `add_bos_token=true`; other
Uncheatable prefix modes use `false`. The Uncheatable-only newline, EOS, and
replacement modes are deliberately not injected into the main conventional
scores, because doing so would create a nonstandard benchmark. On a pilot set,
repeat LAMBADA under the nearest alternate BOS setting and report the maximum
score change as a prefix sensitivity check.

## Statistical analysis

Join results by `model_key`, keep the same complete model cohort for every
comparison, and define `quality = -CR` so all correlations have the same
direction. Report these analyses separately for MMLU, LAMBADA accuracy,
HumanEval pass@1, and HumanEval+ pass@1:

1. Spearman rho and Kendall tau between aggregate Uncheatable quality and each
   conventional score, with family-cluster bootstrap intervals.
2. Category-aligned correlations: scientific-paper CR with MMLU STEM;
   English-fiction CR with LAMBADA; Python-code CR with HumanEval(+). Treat the
   alignments as preregistered primary comparisons and all other category pairs
   as exploratory with false-discovery-rate correction.
3. Residual correlations after fitting each score separately to log10 total
   parameters. Also show within-size-band and within-family estimates where
   enough models exist. This addresses the largest shared confounder.
4. Leave-one-family-out prediction. Compare ridge models using (a) log size,
   (b) aggregate CR, (c) log size plus aggregate CR, and (d) log size plus all
   category CR values. Report held-out R-squared/MAE and the improvement of
   (c)/(d) over (a). Tune ridge regularization only inside each training fold.
5. Rank concordance and top-k overlap as secondary results. Do not average raw
   benchmark scores: MMLU, LAMBADA, and HumanEval have different floors and
   measurement noise. If a composite is needed, average within-cohort z-scores
   and label it secondary.

The strongest support for validity is a category-aligned association that
survives size adjustment and improves leave-one-family-out prediction. A high
raw correlation that disappears after size adjustment only shows that both
benchmarks reward larger models.

## Execution order

Do not begin with all 90 models. First run prompt inspection on two small
checkpoints, then a family-by-size pilot of roughly 25 models: all available
sizes from Qwen2.5, Qwen3, Gemma 3, SmolLM2, Falcon-H1, and Llama/Minitron.
This pilot is large enough to expose loader failures and to estimate whether the
size-adjusted signal is measurable. Only then expand to every working HF model.

MMLU dominates likelihood-scoring cost because each question scores four
continuations. Run MMLU and LAMBADA before HumanEval so the likelihood-task
cohort and loader failures are known. HumanEval may produce many tied zero scores for small general-purpose
base models; keep those models rather than selecting on code ability, and report
the resulting ties and binomial uncertainty. Treat models that fail to load as
missing for a documented technical reason, never as zero-scoring models.

Because MMLU, LAMBADA, and HumanEval are public static benchmarks, they are not
uncontaminated ground truth. Their role is external convergent validation. A
follow-up replication can add HellaSwag/ARC-Challenge and a newer code set such
as LiveCodeBench, but it should not replace the preregistered three-benchmark
analysis after results are seen.

## Backend coverage

Start the paper analysis with the HF cohort and report its coverage explicitly.
For RWKV, add one lm-eval `TemplateLM` adapter implementing `loglikelihood`,
`loglikelihood_rolling`, and `generate_until` around the existing RWKV loader;
the first two enable MMLU/LAMBADA. For HumanEval, iterate over EvalPlus prompts
with the same generation method, write EvalPlus's `{task_id, solution}` JSONL
schema, and run only the scoring stage in Docker. The three
Ministral checkpoints use `Mistral3ForConditionalGeneration` plus
`MistralCommonBackend`, so they need a separate tokenizer/model wrapper rather
than being forced through `AutoModelForCausalLM`. Add these backends only after
matching one HF checkpoint against the reference harness request-by-request.

Primary tool documentation:

- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
- https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu
- https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/lambada
- https://github.com/evalplus/evalplus/blob/master/docs/cli.md
