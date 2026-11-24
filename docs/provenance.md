# Provenance

## What this repository records

A working deployment of Qwen3.8-Flash-Next NVFP4 on four RTX 4090 48GB cards,
built from the PICO-llm 4x4090 deployment branch, with the failures and fixes
that were required to get from "weights load" to "served inference".

The deployment was performed on a PAI DSW container (no systemd, supervisor
based) with:

- 4x RTX 4090 48GB, driver 570.153.02
- 64 CPU threads, 128 GiB RAM
- CUDA 12.8 toolkit at `/usr/local/cuda-12.8`
- vLLM 0.30.0 built from source, torch 2.13.0+cu126

## Timeline of the important failures

1. **Checkpoint download and verification.** 206 files, SHA-256 verified,
   ~126 GiB. The publisher's index and quantized expert layout were used as-is.

2. **FP8 hybrid overlay.** Four BF16 shards were copied and converted in place
   (156 tensors: GDN + QSA side layers). Worst per-tensor relative error 0.0354.
   The source checkpoint was left unchanged. An earlier in-place converter was
   used on the reference host; `scripts/02a-build-fp8hybrid.py` in this repo is
   the non-destructive equivalent with a manifest and the same 156-target
   selection.

3. **CUDA 12.8 toolchain.** The host's CUDA 12.8 install was incomplete for
   source builds. `nvcc`, `cicc`, `ptxas`, `nvlink`, `libcudadevrt.a` and
   `libcudart_static.a` had to be supplied from a full 12.8 toolkit. A CUDA 13
   `ptxas` produced binaries that failed at runtime with
   `CUDA error: invalid resource handle` in the Marlin FP8 path; rebuilding
   with the native 12.8 toolchain fixed it.

4. **FA2 missing.** The vLLM wheel had been built while the
   `vllm-flash-attn` external project was replaced by empty stub targets. Text
   model loading succeeded, then startup died during vision-encoder profiling:

   ```text
   AttributeError: '_OpNamespace' '_vllm_fa2_C' object has no attribute 'varlen_fwd'
   ```

   Building upstream FA2 from `vllm-project/flash-attention` (`506341a`) and
   installing `_vllm_fa2_C.abi3.so` plus `layers/rotary.py` fixed startup.
   The FA2 link needed `LIBRARY_PATH=/usr/local/cuda-12.8/lib64`.

5. **Runtime patches.** FLA shared-memory gate, mamba prefix-cache block size,
   QSA FP8 KV read path, QSA fused-draft metadata, and the hybrid FP8 shim.

6. **Service.** Supervisor service with `startsecs=300`; first `/health` after
   ~10–12 minutes. Four workers, ~43.4 GiB per GPU, KV pool 1,978,617 tokens.

## Verification performed

- `GET /v1/models` returns `qwen38-flash-next-nvfp4`.
- `POST /v1/chat/completions` ("2+2") returns `4`.
- FA2 `varlen_fwd` and `apply_rotary_emb` executed on an SM89 GPU with finite
  outputs.
- Prefill measured at 4K/16K/64K/128K with unique prompts.
- Decode measured single-stream and at 4/8/16 streams.
- KV capacity and per-card memory recorded from the startup log.

Image understanding was **not** validated with a real image; only the encoder
profile run at startup was exercised.

## Credits

This recipe stands on:

- [vLLM](https://github.com/vllm-project/vllm) and
  [vllm-project/flash-attention](https://github.com/vllm-project/flash-attention)
- [PICO4robot/PICO-llm](https://github.com/PICO4robot/PICO-llm), branch
  `qwen3.8-flash-next-nvfp4-4x4090`, for the deployment shape and the 02g patch
- [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) for
  the QSA FP8 KV and mamba cache fixes
- [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)
  for the blockwise-FP8 converter and hybrid shim
- [chenxin199305/qwen38-flash-next-3x4090](https://github.com/chenxin199305/qwen38-flash-next-3x4090)
  for the repository structure and benchmark methodology

See [NOTICE](../NOTICE) for license details.

## AI disclaimer

The scripts, documentation and benchmark harness were assembled by an AI
assistant from the deployment transcript and the referenced repositories. They
were tested on the host described above, not across a matrix of distributions
or driver versions. Treat exact-anchor patches as version-specific and validate
on your own hardware.
