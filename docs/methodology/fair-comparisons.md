# Fair comparisons between serving engines

A comparison between vLLM, SGLang and TensorRT-LLM (or two configurations of one engine)
is only meaningful when everything except the variable under study is held constant.

## Checklist

| Must match | How InferenceLab enforces or records it |
|---|---|
| Same model checkpoint | one `ModelSpec`; deployments reference it by name |
| Same revision | commit-SHA revision pinned in `configs/models/`; recorded in metadata |
| Same GPU (type, count, driver) | same host for all deployments of a campaign; driver/GPU recorded |
| Same precision / quantization | deployment `quantization` + `dtype`; comparisons across formats are separate experiments with quality checks |
| Same context settings | same `max_context`, same max concurrency / batch token budget where the engine exposes them |
| Same caching / chunking policy | `prefix_caching` and `chunked_prefill` set explicitly, not left to differing defaults |
| Same sampling | temperature 0, explicit top_p; vLLM `--generation-config vllm` and SGLang `--sampling-defaults openai` stop checkpoint defaults from silently differing |
| Same workloads | same workload files and seeds |
| Same output lengths | `ignore_eos: true` + fixed `max_tokens` (when the engine supports `ignore_eos`) |
| Same warmup | workload warmup spec |
| Same measurement window | warmup/cooldown phases; makespan duration |
| Same client placement | load generator on the GPU host for every engine |
| Same engine versions | images pinned by tag (+ digest); digest recorded per run |

## Do not

* Compare against **published numbers** from vendors, papers or blog posts — different
  hardware, versions, workloads and measurement code make them incomparable.
* Compare a tuned configuration of one engine with defaults of another. Either tune both
  with the same effort or compare defaults explicitly labelled as defaults.
* Mix quantization formats in a "speed" comparison without a quality check.
* Run the full matrix before the basic steps work (see the GPU runbook's order).
* Draw conclusions from single runs at saturation without repeats; variance at the knee
  of the curve is high.

## Engine-specific caveats (verified 2026-11-13)

* vLLM enables prefix caching and chunked prefill by default; SGLang enables RadixAttention
  but its default schedule policy is now `fcfs`; TensorRT-LLM enables KV block reuse but
  **not** chunked prefill by default. InferenceLab sets these explicitly per deployment.
* vLLM's default `max_num_seqs` / `max_num_batched_tokens` depend on GPU memory; always pin.
* SGLang's `--mem-fraction-static` and vLLM's `--gpu-memory-utilization` do not mean the
  same thing; the deployment notes record which value was used.
* TensorRT-LLM's PyTorch backend cannot load Hugging Face AutoAWQ checkpoints, so an
  "AWQ vs AWQ" comparison including TensorRT-LLM needs a ModelOpt-exported checkpoint.

## Reporting

Comparison tables list only measured values (`n/a` otherwise), include engine versions
and image digests, and keep SIMULATED and GPU results strictly apart.
