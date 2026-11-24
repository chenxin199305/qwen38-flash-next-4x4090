# Troubleshooting

Symptom → cause → fix, in the order these problems appeared on the reference
host.

## Startup

### `'_OpNamespace' '_vllm_fa2_C' object has no attribute 'varlen_fwd'`

```text
WorkerProc hit an exception ...
  File "vllm/model_executor/models/qwen3_vl.py", line ..., in embed_multimodal
AttributeError: '_OpNamespace' '_vllm_fa2_C' object has no attribute 'varlen_fwd'
EngineCore failed to start
```

The wheel has no working FA2 extension. vLLM profiles the vision encoder during
`determine_available_memory()` even for text-only serving, and that profile run
calls FA2. Build it:

```bash
./scripts/02b-build-fa2.sh
```

Verify:

```bash
/root/vllm-venv/bin/python -c "import torch, vllm.vllm_flash_attn as fa; \
print(fa.FA2_AVAILABLE, hasattr(torch.ops._vllm_fa2_C, 'varlen_fwd'))"
```

### `cannot find -lcudadevrt` / `cannot find -lcudart_static`

The FA2 link command lacks a CUDA `-L` path. Export:

```bash
export LIBRARY_PATH=/usr/local/cuda-12.8/lib64
```

`scripts/02-build-vllm.sh` and `02b-build-fa2.sh` already do this.

### `ImportError: libcudart.so.13: cannot open shared object file`

You installed an official vLLM 0.30 wheel on driver 570. Those wheels link
CUDA 13. Build from source with CUDA 12.8 (this repo) or upgrade to driver
>= 580.

### `ModuleNotFoundError: No module named 'vllm.vllm_flash_attn.layers'`

An FA2-less wheel was installed and the Python helpers (`layers/rotary.py`)
were never copied. `02b-build-fa2.sh` installs both the `.so` and the Python
files.

### `CUDA error: invalid resource handle` in `prepare_fp8_layer_for_marlin`

Observed when the FA/extension binaries were compiled by a CUDA 13 `ptxas`
against a driver 570 host. The cubins used an ABI the driver would not load.
Fix: rebuild with the native CUDA 12.8 toolchain (`nvcc`, `cicc`, `ptxas`,
`nvlink` all from `/usr/local/cuda-12.8`), then reinstall the wheel.

### `out of resource: shared memory` in a Triton FLA kernel

SM89 exposes 101,376 bytes of dynamic shared memory; vLLM's FLA default is
102,400. Apply:

```bash
./scripts/03-apply-patches.sh
```

The patch changes `DEFAULT = 102400` to `101376` in
`flash_linear_attention/ops/utils.py`.

### QSA rejects the KV cache dtype

```text
NotImplementedError: Qwen4Exp QSA requires a BF16 main KV cache
```

Apply `patches/02c-qsa-fp8-kv.py` via `03-apply-patches.sh`. It widens the
guards and dequantizes the read path with the layer's real scales.

### Model loads but KV cache is tiny or OOM at load

- Confirm the FP8 overlay is being served (`MODEL_PATH` points at
  `...-fp8hybrid`).
- Confirm `VLLM_PLE_CPU_OFFLOAD=1`; the 51.2 GB PLE table otherwise lands in
  VRAM.
- Reduce `MAX_MODEL_LEN` or `MAX_NUM_SEQS`.
- On 24GB cards this model does not fit; 48GB cards are required.

### First load takes ~9 minutes and RAM is nearly full

126 GiB of safetensors are read, and the PLE table is pinned in host RAM. On a
128 GiB host only ~12 GiB is free during load, so page cache cannot help the
first run. Subsequent starts are faster only if the host has spare RAM. 256 GiB
is recommended.

## Runtime

### Prefix cache changes the answer / restored state is zero

vLLM overwrites `cache_config.block_size` with the minimum block size across KV
groups (the QSA raw-key ring uses 8/16). The mamba state seed then divides by
the wrong block size. Apply `patches/02b-mamba-block-size.py`. Verify with a
repeat-prefix test: greedy output must be identical on a cold and a cached
request.

### MTP gives no speedup; log says fused draft decode is unsupported

```text
Fused multi-step draft decode is not supported ... falling back
```

Apply `patches/02g-qsa-spec-cache.py` (upstream PR #58449 port). The fallback
line should disappear on the next start.

### `Speculative decoding (method=mtp) is enabled but no KV cache group ...`

Harmless warning in 0.30.0. MTP shares the target KV cache in this model; the
benchmark confirms the speedup.

### `Failed to import the DeepSelect extension (vllm._deepselect_C)`

Harmless on SM89. This deployment does not build DeepSelect. QSA falls back to
its Triton path.

### `max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings`

vLLM reserves draft-token slots. If you need more batched tokens, raise
`--max-num-batched-tokens` (set `MAX_NUM_BATCHED_TOKENS` in `.env`) or lower
`MTP_K`.

### `FlashInfer top-p/top-k sampling disabled`

Expected: `VLLM_USE_FLASHINFER_SAMPLER=0` avoids the FlashInfer sampler on this
stack. The native sampler is used.

### Port already in use / service restarts forever

```bash
ss -ltn 'sport = :12345'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

The serve script refuses to start when a selected GPU has a compute process.
Stop the old service (supervisorctl / systemctl / `./stop.sh`) before retrying.

## Service

### PAI DSW / supervisor

```bash
SUP="/etc/dsw/runtime/bin/supervisorctl -c /etc/dsw/sys_configs/supervisor.conf"
$SUP status vllm-tp4
$SUP restart vllm-tp4
tail -n 200 /var/log/vllm-tp4.log
```

`startsecs=300` means supervisor keeps the program in `STARTING` for at least
five minutes; model load takes longer than that, so a `STARTING` status is
normal for ~10 minutes.

### systemd user unit

```bash
systemctl --user status vllm-qwen38-tp4
journalctl --user -u vllm-qwen38-tp4 -n 200
```

If PID 1 is not systemd (containers often use supervisor), use
`./start.sh` / `./stop.sh` instead.

### Collecting a report

```bash
./scripts/doctor.sh > doctor.txt
```

Review the output for keys or private addresses before sharing.
