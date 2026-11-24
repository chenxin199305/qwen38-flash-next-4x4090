# Quick start

This walkthrough assumes a fresh root shell on the reference-class host:
4x RTX 4090 48GB, driver 570.153.02, CUDA 12.8 toolkit installed under
`/usr/local/cuda-12.8`, 128 GiB RAM (256 GiB recommended).

## 1. Clone and configure

```bash
git clone https://github.com/chenxin199305/qwen38-flash-next-4x4090.git
cd qwen38-flash-next-4x4090
cp .env.sample .env
$EDITOR .env
```

Minimum edits:

```bash
MODEL_DIR="$HOME/models/Qwen3.8-Flash-Next-NVFP4"
FP8HYBRID_DIR="$HOME/models/Qwen3.8-Flash-Next-NVFP4-fp8hybrid"
MODEL_PATH="$FP8HYBRID_DIR"
GPU_IDS="0,1,2,3"
PORT=12345
```

Keep `BIND_HOST=127.0.0.1` unless you configure an API key.

## 2. Preflight

```bash
./scripts/00-preflight.sh
```

It checks four 46GB GPUs, driver, RAM, free disk, `nvcc`, the CUDA static
libraries, the port and the checkpoint layout. Fix anything it flags before
continuing.

## 3. Install

```bash
./install.sh
```

This runs, in order:

1. `scripts/01-setup-venv.sh` — venv + torch 2.13.0+cu126 + build tools.
2. `scripts/02-build-vllm.sh` — vLLM 0.30.0 source wheel **including FA2**.
3. `scripts/02a-build-fp8hybrid.py` — 156-tensor blockwise-FP8 overlay.
4. `scripts/03-apply-patches.sh` — FLA shmem, mamba cache, QSA FP8 KV,
   QSA fused-draft metadata, hybrid FP8 shim.
5. `scripts/04-install-service.sh` — supervisor (DSW/PAI) or systemd user unit.

Expect 30–90 minutes for the vLLM build; the wheel is ~150 MB and the FA2
extension is ~470 MB before stripping.

Skip steps you already completed:

```bash
./install.sh --skip-venv --skip-build
```

## 4. Start and verify

If a service was installed:

```bash
# PAI DSW / supervisor
/etc/dsw/runtime/bin/supervisorctl -c /etc/dsw/sys_configs/supervisor.conf status vllm-tp4

# or systemd
systemctl --user status vllm-qwen38-tp4
```

Otherwise:

```bash
./start.sh --wait          # ~8-12 minutes for the first load
./scripts/healthcheck.sh
./scripts/smoke.sh
```

`healthcheck.sh` prints `/health`, the model id and per-GPU memory.
`smoke.sh` asks `2+2` and checks that the answer contains `4`.

## 5. Call the API

```bash
curl http://127.0.0.1:12345/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen38-flash-next-nvfp4",
    "messages": [{"role": "user", "content": "用一句话解释 MoE。"}],
    "max_tokens": 128
  }'
```

With an API key:

```bash
echo 'your-key' > ~/.vllm-key && chmod 600 ~/.vllm-key
# set API_KEY_FILE="$HOME/.vllm-key" in .env, then restart
curl -H "Authorization: Bearer $(cat ~/.vllm-key)" ...
```

## 6. Benchmark

```bash
python benchmarks/bench.py --api-key-file ~/.vllm-key --runs 3 \
    --max-tokens 256 --ignore-eos
python benchmarks/bench_perf.py --api-key-file ~/.vllm-key
python benchmarks/bench-concurrency.py --api-key-file ~/.vllm-key \
    --levels 1,4,8,16 --max-tokens 256 --ignore-eos
```

Important: prefill benchmarks must use a **fresh random prompt** for every
request. Reusing a prompt measures the prefix cache (hundreds of thousands of
tok/s), not the GPU.

## First-boot expectations

| Event | Time / value |
|---|---|
| Target model weights | ~9 minutes first load, ~1 minute from page cache |
| MTP draft scan | ~40 seconds |
| CUDA graph capture | ~1–2 minutes |
| `/health` ready | ~10–12 minutes after start |
| Model load per card | 23.99 GiB |
| KV cache per card | 14.54 GiB |
| KV pool | 1,978,617 tokens |

The first start is slow because 126 GiB of safetensors are read and the FA2
kernels are exercised for the vision profile. Later starts benefit from page
cache if the host has spare RAM.

## Where things go

| Path | Contents |
|---|---|
| `.env` | local configuration (git-ignored) |
| `build/` | wheels, dependency checkouts, backups (git-ignored) |
| `$LOG_DIR/vllm.log` | background `start.sh` log |
| `/var/log/vllm-tp4.log` | supervisor log on PAI DSW |
| `~/.local/state/qwen38-flash-next/` | default `LOG_DIR` outside DSW |
