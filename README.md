# Qwen3.8-Flash-Next NVFP4 on 4x RTX 4090 48GB

**TP4, 524K context, 1.98M KV tokens, 163.8 tok/s single-stream, 1,244.8 tok/s aggregate at 16 streams, ~7.2K tok/s prefill.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-0.30.0-green.svg)](https://github.com/vllm-project/vllm)
[![GPU](https://img.shields.io/badge/GPU-4x%20RTX%204090%2048GB-orange.svg)](#requirements)

> **AI-generated content.** The scripts and notes in this repository were
> assembled by an AI assistant from a real deployment process. They may contain
> errors or omit environment-specific details. Issues and corrections are
> welcome. The repository does not distribute model weights.

## Why this recipe

Four 48GB cards let the model run with plain tensor parallelism and no
expert-parallel workarounds. The hard part is not the GPU count — it is getting
a CUDA 12.8 vLLM build that actually contains the FlashAttention-2 extension.

| Advantage | How | Result |
|---|---|---|
| **All four cards used** | TP4 splits attention and dense layers; NVFP4 experts are sharded by TP | One 524K-context server, no EP padding |
| **512K context with concurrency** | YaRN factor 2.0 + FP8 KV | KV pool **1,978,617 tokens** = **3.77x** a 524K request |
| **Small model footprint** | Blockwise-FP8 side layers | **23.99 GiB/card** model load, **14.54 GiB/card** KV |
| **Useful single-stream speed** | MTP k=2 + QSA fused-draft metadata | **163.8 tok/s** decode |
| **Useful aggregate throughput** | 32 scheduler slots | **425.8 / 735.3 / 1,244.8 tok/s** at 4/8/16 streams |
| **Fast long-prompt ingest** | Chunked prefill, 4,096-token threshold | **7.0–7.3K tok/s** from 4K to 131K prompts |
| **PLE table does not eat VRAM** | vLLM pinned CPU offload (`VLLM_PLE_CPU_OFFLOAD=1`) | 51.2 GB n-gram table lives in host RAM |
| **Boots past the vision profile** | FA2 built from `vllm-project/flash-attention` | No `_vllm_fa2_C.varlen_fwd` crash |
| **Reproducible** | vLLM 0.30.0 pinned, patch `--check`, benchmark JSON | Every claim has a command and a recorded config |

## Requirements

| Item | Requirement |
|---|---|
| GPU | **4x RTX 4090 48GB** (reports 46,068 MiB). 24GB cards cannot hold the model. |
| CPU / RAM | x86_64, **256 GiB recommended**; 128 GiB works but is tight (PLE is ~51 GiB pinned host RAM). |
| Disk | ~200 GiB free (126 GiB base + 28 GiB overlay + wheel/build) |
| OS | Linux, no systemd required |
| Driver | **570.153.02** (CUDA 12.8). Driver >= 580 can use newer wheels. |
| CUDA toolkit | **12.8 full toolkit** with `nvcc`, `cicc`, `ptxas`, `nvlink`, `libcudadevrt.a`, `libcudart_static.a` |
| vLLM | **0.30.0** — patch anchors are version-specific |
| torch | 2.13.0+cu126 |
| Model | Qwen3.8-Flash-Next NVFP4 checkpoint: 48 layers, 512 experts, `moe_intermediate_size=640`, 1 MTP layer, ~126 GiB |
| Model access | Obtain a compatible checkpoint yourself; this repository does not distribute weights. |

> **Checkpoint license.** The checkpoint is governed by its own license (see
> the model card). This repository does not redistribute weights or generated
> overlays. Obtain the checkpoint from its publisher and comply with its terms.

## Quick start

```bash
git clone https://github.com/chenxin199305/qwen38-flash-next-4x4090.git
cd qwen38-flash-next-4x4090

cp .env.sample .env
# Edit MODEL_DIR / FP8HYBRID_DIR / GPU_IDS / PORT.
# Keep BIND_HOST=127.0.0.1 unless you also configure an API key.

./install.sh
./start.sh --wait
./scripts/healthcheck.sh
./scripts/smoke.sh
```

Already have a working vLLM 0.30 venv and the NVFP4 checkpoint?

```bash
./install.sh --skip-venv --skip-build
```

Rebuild only the missing FA2 extension into an existing venv:

```bash
./scripts/02b-build-fa2.sh
```

Minimal API call:

```bash
curl http://127.0.0.1:12345/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-nvfp4","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Architecture

```text
OpenAI-compatible client
          |
          v
  vLLM 0.30.0  TP4  (GPU 0/1/2/3)
          |
          |-- NVFP4 routed experts (512 experts, top-10)
          |-- FP8 hybrid side layers: GDN in/out, QSA q/k/v/o (156 tensors)
          |-- FP8 KV: 1,978,617 tokens at 524K context
          |-- MTP=2 with QSA fused-draft metadata
          |-- FA2 (SM80+PTX kernels) for full attention and the vision encoder
          '-- PLE 51.2 GB pinned host RAM (UVA)
```

- **TP4** splits attention / dense layers and the NVFP4 expert weights across
  four GPUs. No expert parallelism, no MARLIN-EP workaround.
- **FP8 hybrid side layers** convert only the dense side tensors (156) to
  blockwise FP8-e4m3; routed experts stay NVFP4.
- **PLE** is an n-gram embedding table. It is a lookup, not a matmul; keeping it
  in pinned host memory leaves VRAM for KV.
- **FA2** is required even for a text-only server: vLLM profiles the vision
  encoder at startup, and that profile run calls `_vllm_fa2_C.varlen_fwd`.
- **FP8 KV + YaRN** is what makes 524K x 3.77 concurrency possible on 48GB cards.

More detail: [docs/architecture.md](docs/architecture.md).

## Deployment details that matter

These are the problems that took the longest to solve. The full symptom →
cause → fix table is in [docs/troubleshooting.md](docs/troubleshooting.md).

| Problem | What you see | Root cause | Fix |
|---|---|---|---|
| Startup dies after loading weights | `AttributeError: '_OpNamespace' '_vllm_fa2_C' object has no attribute 'varlen_fwd'` | Source wheel was built while the `vllm-flash-attn` external project was stubbed out; the vision profile run still needs FA2 | `scripts/02-build-vllm.sh` includes the FA subproject; `scripts/02b-build-fa2.sh` repairs an existing venv |
| FA2 link fails | `cannot find -lcudadevrt` / `-lcudart_static` | CUDA static libs are not on the linker search path | Export `LIBRARY_PATH=$CUDA_HOME/lib64` (the build script does) |
| PyPI wheel import fails on driver 570 | `libcudart.so.13` / version errors | vLLM 0.30 wheels link CUDA 13 | Build from source with CUDA 12.8; driver >= 580 avoids this |
| FP8 KV rejected | Startup assertion from QSA | QSA read path and guards assume BF16 | `patches/02c-qsa-fp8-kv.py` |
| Prefix cache returns wrong output | Greedy output changes on a cache hit | Mamba state slot seeded with the minimum KV block size | `patches/02b-mamba-block-size.py` |
| MTP gives no speedup | Draft metadata rebuilt every step | QSA backend lacks fused-draft metadata support | `patches/02g-qsa-spec-cache.py` (upstream PR #58449) |
| Model does not fit / KV too small | OOM or tiny KV pool | BF16 side layers | `scripts/02a-build-fp8hybrid.py` converts 156 tensors to blockwise FP8 |
| PLE table eats VRAM | OOM at load | 51.2 GB table treated as GPU memory | `VLLM_PLE_CPU_OFFLOAD=1` pinned host memory |
| FLA kernel launch fails | `out of resource: shared memory` | FLA default 102400 B > SM89 101376 B | `patches/02a-fla-shmem.py` |
| YaRN silently ignored | Context still 262K | `--rope-scaling` removed in newer vLLM | Use `--hf-overrides` with `text_config.rope_parameters` |
| 24GB 4090 fails | OOM at load | Model needs ~24 GiB/card plus KV | 48GB cards required |

## Measured performance

Reference host: 4x RTX 4090 48GB, driver 570.153.02, vLLM 0.30.0,
torch 2.13.0+cu126, 128 GiB RAM. Token counts come from
`usage.completion_tokens`, not SSE chunks (MTP emits several tokens per chunk).

### Decode (single stream)

| Run | Prompt tokens | Completion tokens | TTFT | Decode |
|---|---:|---:|---:|---:|
| 1 | 87 | 243 | 0.050s | 163.8 tok/s |
| 2 | 87 | 218 | 0.049s | 165.5 tok/s |
| 3 | 87 | 237 | 0.049s | 153.5 tok/s |
| **Median** | | | | **163.8 tok/s** |

### Prefill (unique prompt, no prefix-cache hit)

| Prompt tokens | TTFT | Prefill rate |
|---:|---:|---:|
| 4,184 | 0.570s | 7,342 tok/s |
| 16,543 | 2.279s | 7,260 tok/s |
| 65,542 | 9.102s | 7,201 tok/s |
| 131,039 | 18.550s | 7,064 tok/s |

### Concurrency (fixed 256 output tokens, `ignore_eos`)

| Streams | Aggregate | Per-stream | TTFT range |
|---:|---:|---:|---:|
| 1 | 155.7 tok/s | 161.4 tok/s | 0.057s |
| 4 | 425.8 tok/s | 112.7–134.6 tok/s | 0.044–0.414s |
| 8 | 735.3 tok/s | 96.2–113.9 tok/s | 0.122–0.464s |
| 16 | **1,244.8 tok/s** | 79.5–91.3 tok/s | 0.054–0.188s |

### KV capacity

| Metric | Value |
|---|---:|
| Model load per card | 23.99 GiB |
| KV cache per card | 14.54 GiB |
| KV pool | 1,978,617 tokens |
| Concurrency at 524K/request | 3.77x |

Methodology and raw JSON: [docs/benchmarks.md](docs/benchmarks.md),
[`benchmarks/results/2026-11-24-4x4090.json`](benchmarks/results/2026-11-24-4x4090.json).

## Tuning profiles

| Profile | MTP | Context | KV dtype | Max seqs | Use case |
|---|---:|---:|---|---:|---|
| `balanced.env` | 2 | 524K | FP8 | 16 | Small-team default |
| `capacity.env` | 0 | 524K | FP8 | 16 | Longest context per request, max KV |
| `throughput.env` | 2 | 524K | FP8 | 32 | Many concurrent agent sessions (tested) |
| `native-262k.env` | 2 | 262K | FP8 | 32 | No YaRN; more KV headroom |

Append a profile to `.env` (later values win):

```bash
cat config/profiles/capacity.env >> .env
```

## Repository layout

```text
install.sh                  one-shot installer
start.sh / stop.sh          background lifecycle (no systemd required)
check-weights.sh            verify base checkpoint + FP8 overlay
.env.sample                 all knobs
config/profiles/            runtime profiles
patches/                    FLA shmem, mamba cache, QSA FP8 KV, QSA draft, hybrid shim
scripts/                    preflight, venv, vLLM build, FA2 repair, overlay,
                            patches, service, serve, health, smoke, doctor
services/                   supervisor + systemd templates
benchmarks/                 single-stream, concurrency, prefill/decode, results
docs/                       quickstart, build, architecture, benchmarks, troubleshooting
```

## Security

- Default bind is `127.0.0.1`.
- A non-loopback bind without an API key is refused unless `ALLOW_INSECURE=1`.
- Keep `.env` and the key file at mode `600`.
- Never paste keys, IPs or logs with credentials into an issue.
- For LAN or public access, put the server behind a TLS reverse proxy with
  authentication. This repository deliberately contains no tunnel or FRP config.

## Known limitations

- **Text verified.** The FA2-based vision encoder initializes and profiles
  correctly, but image understanding has not been validated end-to-end.
- **48GB cards only.** 4x24GB cannot hold ~24 GiB/card plus KV.
- **vLLM 0.30.0 only.** Patches are exact-anchor edits; another version fails
  `--check` by design.
- **vLLM 0.33 not used.** Its official wheels chain CUDA 13, which normally
  requires driver >= 580; this host runs 570.153.02. The PICO-llm branch that
  this deployment follows was validated on 0.30.
- **supervisor and systemd paths are both provided**; the supervisor path is the
  one exercised on the reference host (PAI DSW container).
- **Throughput numbers are one host, one run.** Re-measure on your cards.
- **MTP k=2 tested.** k=0/1/3 are accepted by `serve.sh` but not re-measured.

## Documentation

| Document | Contents |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | Step-by-step install and first request |
| [docs/build.md](docs/build.md) | CUDA 12.8 vLLM + FA2 build, flags and failure modes |
| [docs/architecture.md](docs/architecture.md) | TP4, FP8 hybrid, PLE, FP8 KV, MTP |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Symptom → cause → fix |
| [docs/benchmarks.md](docs/benchmarks.md) | Methodology and raw results |
| [docs/compatibility.md](docs/compatibility.md) | Driver / vLLM / torch matrix |
| [docs/provenance.md](docs/provenance.md) | Source, credits, AI disclaimer |

## License and credits

Code: [Apache-2.0](LICENSE). Model weights: their own license, not included.

The runtime patches build on work from
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX),
[Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound),
[PICO4robot/PICO-llm](https://github.com/PICO4robot/PICO-llm), and the sibling
[3x4090 recipe](https://github.com/chenxin199305/qwen38-flash-next-3x4090).
See [NOTICE](NOTICE) for details.
