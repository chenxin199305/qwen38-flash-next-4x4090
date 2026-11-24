# Building vLLM 0.30.0 for driver 570 / CUDA 12.8

The reference host runs driver `570.153.02`. vLLM 0.30's official wheels link
CUDA 13 and fail at import on this driver (`libcudart.so.13` unresolved or a
CUDA version error). The working path is a source build against CUDA 12.8 with
torch 2.13.0+cu126.

## Toolchain

| Component | Verified value |
|---|---|
| Driver | 570.153.02 |
| CUDA toolkit | 12.8 at `/usr/local/cuda-12.8` |
| torch | 2.13.0+cu126 |
| Python | 3.11 |
| GCC | 11.x |
| Rust | stable (rustup) |
| CMake | >= 3.26 |
| Ninja | any recent |
| RAM during build | 32 GiB+ free recommended |

The CUDA 12.8 installation must include the compiler and static runtime, not
just the runtime:

```bash
ls /usr/local/cuda-12.8/bin/{nvcc,cicc,ptxas,nvlink}
ls /usr/local/cuda-12.8/lib64/{libcudadevrt.a,libcudart_static.a}
```

A runtime-only or partially unpacked toolkit produces link failures such as
`cannot find -lcudadevrt`.

## The FA2 trap

vLLM 0.30's `CMakeLists.txt` normally pulls `vllm-flash-attn` from GitHub via
`cmake/external_projects/vllm_flash_attn.cmake`. In the deployment this recipe
is based on, that include had been replaced with empty stub targets to avoid a
throttled GitHub route. The resulting wheel imported fine and served text, but
the vision-encoder profile run at startup died with:

```text
AttributeError: '_OpNamespace' '_vllm_fa2_C' object has no attribute 'varlen_fwd'
```

FA2 is needed **even for a text-only server**, because vLLM profiles the
multimodal encoder during `determine_available_memory()`. The fix is to build
the upstream FA subproject, either as part of the wheel or afterwards:

```bash
# part of a fresh build
./scripts/02-build-vllm.sh

# repair an existing venv
./scripts/02b-build-fa2.sh
```

`02b` configures a separate CMake build directory, builds only target
`_vllm_fa2_C`, and copies the extension plus `layers/rotary.py` into the
installed `vllm/vllm_flash_attn/` package. It is idempotent: if
`torch.ops._vllm_fa2_C.varlen_fwd` exists, it exits immediately.

FA2 is built for `8.0+PTX`, not `8.9`: upstream FA2 has no SM89-specific
instantiation. The SM80 kernels run correctly on the 4090 and PTX provides
forward compatibility. A smoke test is included in `02b`'s verification step.

## What `02-build-vllm.sh` does

1. Clones `vllm-project/vllm` at tag `v0.30.0`.
2. Clones `vllm-project/flash-attention` at commit `506341a` and initializes
   its CUTLASS submodule.
3. Clones CUTLASS `v4.7.1` and Triton `v3.5.1` into `build/deps/` and points
   `VLLM_CUTLASS_SRC_DIR` / `TRITON_KERNELS_SRC_DIR` at them.
4. Re-inserts the `vllm_flash_attn.cmake` include after the `triton_kernels`
   include (the FA project must win macro collisions and must be last).
5. Runs `setup.py bdist_wheel` with:

```bash
CUDA_HOME=/usr/local/cuda-12.8
LIBRARY_PATH=/usr/local/cuda-12.8/lib64
VLLM_FLASH_ATTN_SRC_DIR=~/src/vllm-flash-attn
TORCH_CUDA_ARCH_LIST=8.9
MAX_JOBS=32
VLLM_TARGET_DEVICE=cuda
```

`LIBRARY_PATH` is not optional: without it the FA2 link step cannot find
`libcudadevrt.a` / `libcudart_static.a` because the generated link command does
not add a CUDA `-L` directory.

6. Installs the wheel. If the resolver replaces torch with a CUDA 13 build,
   the script restores the cu126 stack and re-verifies.
7. Verifies `import vllm`, FA2 availability and the `varlen_fwd` op.

## Build time and artifacts

| Artifact | Approximate size / time |
|---|---|
| vLLM wheel | ~150 MB |
| FA2 extension (unstripped) | ~470 MB |
| Full build on 64 cores / 128 GiB | 30–60 minutes |
| FA2-only rebuild | 8–15 minutes |

Object files for the main vLLM build can exceed 60 GiB. Leave at least 100 GiB
free under `build/` and `dist/`.

## Why not vLLM 0.33

vLLM 0.33's official wheels chain CUDA 13, which normally requires driver
>= 580. This host has 570.153.02, and the PICO-llm branch this deployment
follows (`qwen3.8-flash-next-nvfp4-4x4090`) was validated against 0.30.0.
Upgrading would require either a driver upgrade or a fresh CUDA 12.8 source
build plus re-validating every patch anchor. That is a separate project, not a
drop-in bump.

## Verifying a build

```bash
/root/vllm-venv/bin/python - <<'PY'
import torch, vllm
import vllm.vllm_flash_attn as fa
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("FA2", fa.FA2_AVAILABLE, "varlen_fwd", hasattr(torch.ops._vllm_fa2_C, "varlen_fwd"))
PY
```

A GPU smoke test of the actual kernel:

```bash
CUDA_VISIBLE_DEVICES=0 /root/vllm-venv/bin/python - <<'PY'
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func
q = torch.randn(128, 8, 64, dtype=torch.float16, device="cuda")
k = torch.randn(128, 8, 64, dtype=torch.float16, device="cuda")
v = torch.randn(128, 8, 64, dtype=torch.float16, device="cuda")
cu = torch.tensor([0, 128], dtype=torch.int32, device="cuda")
out = flash_attn_varlen_func(q, k, v, 128, cu, 128, cu_seqlens_k=cu, causal=True)
torch.cuda.synchronize()
print(out.shape, torch.isfinite(out).all().item())
PY
```
