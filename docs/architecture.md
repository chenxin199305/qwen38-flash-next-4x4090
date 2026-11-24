# Architecture

## Model shape

| Property | Value |
|---|---|
| Architecture | `Qwen4ExpForConditionalGeneration` (Qwen3-VL text stack + Qwen3-Next-style hybrid layers) |
| Layers | 48 (interleaved GDN linear attention / QSA full attention) |
| Hidden size | 2,560 |
| Attention heads | 24 (2 KV heads) |
| Routed experts | 512, top-10 |
| MoE intermediate | 640 |
| Vocab | 248,320 |
| Native context | 262,144 |
| Extended context | 524,288 (YaRN factor 2.0) |
| MTP layers | 1 |
| Quantization | NVFP4 routed experts; BF16 side layers converted to blockwise FP8 |

## Why TP4 without EP

Four 48GB cards provide enough aggregate memory that expert parallelism is not
needed. TP4 divides attention, dense layers and expert weights across the four
GPUs. This avoids the EP-specific code paths and fixes that a TP3+EP3 recipe
needs (no MARLIN EP top-k workaround, no expert map).

The trade-off is that each GPU holds a slice of every expert, and the
all-reduce traffic is higher than with EP. On four PCIe 4090s the measured
decode and prefill rates are still good (see [benchmarks](benchmarks.md)).

## Hybrid FP8 side layers

The checkpoint quantizes only the routed experts to NVFP4. Dense side layers
remain BF16 and are read in full on every decoded token:

- GDN `in_proj_qkv`, `in_proj_z`, `out_proj` (3 × 36 layers)
- QSA `q_proj`, `k_proj`, `v_proj`, `o_proj` (4 × 12 layers)

That is 156 tensors. `scripts/02a-build-fp8hybrid.py` converts them to
DeepSeek-style blockwise FP8-e4m3:

```text
weight           float8_e4m3fn  [out, in]
weight_scale_inv float32        [out/128, in/128]
```

The base checkpoint is never modified: unchanged files are symlinked, only the
four shards containing target tensors are rewritten. Worst per-tensor relative
error measured on the reference host: **0.0354**.

At runtime, `patches/vllm_fp8_hybrid_modelopt.py` scans the checkpoint metadata
for `F8_E4M3` weights with a `weight_scale_inv` sibling and routes those linear
modules to vLLM's generic blockwise `Fp8Config`, while the NVFP4 MoE path stays
on `ModelOptNvFp4Config`.

## PLE in host RAM

The model has a 51.2 GB n-gram embedding (PLE) table. It is a lookup, not a
matmul. `VLLM_PLE_CPU_OFFLOAD=1` keeps it in pinned host memory; each token
performs eight lookups over UVA. This is why the host needs 128 GiB minimum
(256 GiB recommended) and why VRAM stays available for KV.

## FP8 KV and YaRN

- Main KV cache: `fp8_e4m3`.
- QSA read path dequantizes with vLLM's `_cast_kv_tile` using the layer's real
  scales (`patches/02c-qsa-fp8-kv.py`).
- QSA indexer caches keep their own dtype knob and are not touched by that
  patch.
- 524K context uses YaRN factor 2.0 via `--hf-overrides`; `--rope-scaling` is
  no longer honored by newer vLLM.

Result on the reference host: **14.54 GiB KV/card**, **1,978,617 tokens**,
**3.77x** a 524K request.

## MTP and QSA draft metadata

`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` uses the
checkpoint's MTP layer as a draft model. The QSA attention backend in 0.30.0
does not declare `supports_draft_decode_metadata_update`, so the speculator
rebuilds attention metadata on every draft step. `patches/02g-qsa-spec-cache.py`
ports upstream PR #58449, which adds in-place update of the draft metadata.

Measured effect on the reference host: single-stream decode ~164 tok/s and
1,244.8 tok/s aggregate at 16 streams. MTP=0 remains a supported capacity
profile.

## FA2's role

FA2 is used for:

- full-attention layers that select the `FLASH_ATTN` backend, and
- the vision encoder's profile run at startup.

The QSA path uses its own Triton kernels and does not consume FA2. This is why
a text-only server can appear to work until the encoder profile run — that is
the first place `_vllm_fa2_C.varlen_fwd` is called.

FA2 is built for `8.0+PTX`. There is no SM89-specific FA2 instantiation; the
SM80 kernels run on the 4090 and PTX covers forward compatibility.

## Memory budget (reference host)

| Item | Per card |
|---|---:|
| Model weights | 23.99 GiB |
| KV cache | 14.54 GiB |
| CUDA graphs / activations / workspace | remainder of 46,068 MiB |
| Total process RSS observed | ~43.4 GiB |
