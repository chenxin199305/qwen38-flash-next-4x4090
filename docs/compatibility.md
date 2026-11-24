# Compatibility

## Verified stack

| Component | Verified value |
|---|---|
| GPU | 4x RTX 4090 48GB (46,068 MiB reported) |
| Driver | 570.153.02 |
| CUDA toolkit | 12.8 (`/usr/local/cuda-12.8`) |
| torch | 2.13.0+cu126 |
| torchvision / torchaudio | 0.28.0+cu126 / 2.11.0+cu126 |
| vLLM | 0.30.0, source build `+cu128` |
| FA2 | `vllm-project/flash-attention` `506341a`, built for `8.0+PTX` |
| Python | 3.11 |
| protobuf | 6.33.5 |
| RAM | 128 GiB (works, tight); 256 GiB recommended |

## Driver / wheel matrix

| Driver | Approach | Status |
|---|---|---|
| 570.x | Source build with CUDA 12.8 (this repo) | **Verified** |
| >= 580 | Official vLLM 0.30 wheel (CUDA 13) | Possible, but the runtime patches must be re-validated |
| 570.x + official 0.30 wheel | — | Fails: wheel links CUDA 13 |
| 570.x + vLLM 0.33 wheel | — | Not attempted; 0.33 wheels chain CUDA 13 |

The PICO-llm branch this deployment follows was validated on vLLM 0.30.0.
vLLM 0.33 is a separate validation project: new patch anchors, new CUDA
requirements, new FA2 build.

## GPU count

| Cards | Recipe |
|---|---|
| 4x 48GB | **This repository**: TP4 |
| 3x 48GB | [qwen38-flash-next-3x4090](https://github.com/chenxin199305/qwen38-flash-next-3x4090): TP3 + EP3 |
| 2x 48GB | Not covered; expert padding and KV become tight |
| 4x 24GB | Does not fit (~24 GiB/card weights before KV) |

## Checkpoint layout

The scripts expect the published NVFP4 layout:

- `config.json` with `Qwen4ExpForConditionalGeneration`
- 48 layers, hidden 2,560, 24 heads, 2 KV heads
- 512 experts, top-10, `moe_intermediate_size=640`
- 1 MTP layer
- `model.safetensors.index.json` and `model-bf16-*.safetensors` shards
- Side layers stored as BF16 (the overlay converter needs them in BF16)
- Routed experts stored as NVFP4 / ModelOpt

If `scripts/02a-build-fp8hybrid.py --dry-run` reports a target count other than
156, the checkpoint layout differs. Stop and inspect before converting.

## Patch anchor policy

All patches are exact-string replacements against vLLM 0.30.0. They fail loudly
when an anchor moves. `03-apply-patches.sh --check-only` reports the current
state without modifying anything. Do not run these patches against another
vLLM version.

## Known harmless warnings

- `Failed to import the DeepSelect extension (vllm._deepselect_C)`
- `Speculative decoding (method=mtp) is enabled but no KV cache group ...`
- `Using default MoE config. Performance might be sub-optimal!` (no tuned
  Triton config for `E=512,N=160` on the 4090)
- `FlashInfer top-p/top-k sampling disabled`
- `Unrecognized keys in rope_parameters ... mrope_section` from transformers

These do not prevent startup or change the measured performance materially.
