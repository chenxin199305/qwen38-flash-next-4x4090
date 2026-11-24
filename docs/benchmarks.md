# Benchmarks

## Method

- Hardware: 4x RTX 4090 48GB (46,068 MiB), driver 570.153.02, 64 CPU threads,
  128 GiB RAM.
- Software: vLLM 0.30.0 (+cu128 source build), torch 2.13.0+cu126, CUDA 12.8,
  FA2 from `vllm-project/flash-attention` `506341a`, built for `8.0+PTX`.
- Model: Qwen3.8-Flash-Next NVFP4 + 156-tensor FP8 overlay.
- Server: TP4, 524K context, FP8 KV, prefix caching, MTP k=2,
  `max-num-seqs=32`, `gpu-memory-utilization=0.95`, async scheduling,
  `watermark=0.05`, `long-prefill-token-threshold=4096`.
- Token counts: `usage.completion_tokens`, **not** SSE chunks. With speculative
  decoding one chunk can contain several tokens; chunk counting measures
  steps, not tokens.
- TTFT: request start to the first content/reasoning chunk.
- Decode rate: completion tokens divided by (last token time − first token
  time).

## Decode, single stream

Three runs, variable EOS, short prompt (~87 tokens).

| Run | Completion tokens | TTFT | Decode |
|---|---:|---:|---:|
| 1 | 243 | 0.050s | 163.8 tok/s |
| 2 | 218 | 0.049s | 165.5 tok/s |
| 3 | 237 | 0.049s | 153.5 tok/s |
| **Median** | | | **163.8 tok/s** |

## Prefill

Unique random prompt per request, `ignore_eos`, `max_tokens=1`, one warmup at
the same size with a different random text. Every measured prompt is fresh, so
no prefix-cache hit is possible.

| Prompt tokens | TTFT | Prefill rate |
|---:|---:|---:|
| 4,184 | 0.570s | 7,342 tok/s |
| 16,543 | 2.279s | 7,260 tok/s |
| 65,542 | 9.102s | 7,201 tok/s |
| 131,039 | 18.550s | 7,064 tok/s |

The rate is nearly flat from 4K to 131K, which is what chunked prefill with a
4,096-token threshold should look like on this hardware.

## Concurrency

Fixed 256 output tokens per stream, `ignore_eos`, separate HTTP requests.
Aggregate = total completion tokens / wall time.

| Streams | Aggregate | Per-stream | TTFT range | Wall |
|---:|---:|---:|---:|---:|
| 1 | 155.7 tok/s | 161.4 tok/s | 0.057s | 1.644s |
| 4 | 425.8 tok/s | 112.7–134.6 tok/s | 0.044–0.414s | 2.405s |
| 8 | 735.3 tok/s | 96.2–113.9 tok/s | 0.122–0.464s | 2.785s |
| 16 | **1,244.8 tok/s** | 79.5–91.3 tok/s | 0.054–0.188s | 3.290s |

At 16 streams the per-stream rate drops but aggregate keeps rising; the
scheduler limit is 32, so more concurrency is possible. Do not quote the
16-stream aggregate as per-user speed.

## KV capacity

| Metric | Value |
|---|---:|
| Model load / card | 23.99 GiB |
| KV cache / card | 14.54 GiB |
| KV pool | 1,978,617 tokens |
| 524,288-token requests in flight | 3.77x |
| GPU process RSS / card | ~43.4 GiB |

## Reproduce

```bash
# decode, 3 runs after warmup
python benchmarks/bench.py --api-key-file ~/.vllm-key --runs 3 \
    --max-tokens 256 --ignore-eos

# prefill + decode with unique prompts
python benchmarks/bench_perf.py --api-key-file ~/.vllm-key \
    --prefill-sizes 4096,16384,65536,131072 \
    --decode-runs 3 --max-tokens 256

# concurrency sweep, JSON output
python benchmarks/bench-concurrency.py --api-key-file ~/.vllm-key \
    --levels 1,4,8,16 --max-tokens 256 --ignore-eos \
    --output benchmarks/results/$(date +%F)-4x4090.json
```

The recorded result is
[`benchmarks/results/2026-11-24-4x4090.json`](../benchmarks/results/2026-11-24-4x4090.json).

## What not to compare

- Do not compare an SSE-chunk count with `usage.completion_tokens` under MTP.
- Do not compare a 262K native run with a 524K YaRN run without labeling the
  context; KV per request differs.
- Do not compare a cold request with a prefix-cache hit. Reusing the same long
  prompt produces six-figure tok/s prefill numbers that say nothing about the
  GPU.
- Do not compare a single run on a different driver/CUDA stack; the FA2 build
  and KV dtype are part of the result.
