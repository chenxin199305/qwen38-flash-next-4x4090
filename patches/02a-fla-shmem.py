#!/usr/bin/env python3
"""Raise the Flash-Linear-Attention shared-memory gate for SM89.

vLLM's FLA kernels default to a 102400-byte dynamic shared-memory request.
SM89 (RTX 4090) exposes 101376 bytes, so the Triton launch fails with
"out of resource: shared memory". Lower the default to the SM89 limit.

Usage:
    python 02a-fla-shmem.py --vllm-root <path/to/site-packages/vllm> [--check]
"""
import argparse
import os
import sys

MARKER = "spark-fla-shmem"
OLD = "DEFAULT = 102400"
NEW = "DEFAULT = 101376  # spark-fla-shmem: sm89 99KiB cache"


def resolve_root(explicit: str) -> str:
    if explicit:
        return explicit
    import vllm
    return os.path.dirname(vllm.__file__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", default="")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = os.path.join(
        resolve_root(args.vllm_root),
        "third_party/flash_linear_attention/ops/utils.py",
    )
    src = open(path).read()
    patched = MARKER in src
    if args.check:
        print(f"{path}: {'patched' if patched else 'NOT patched'}")
        return 0 if patched else 1
    if patched:
        print("FLA shmem already patched")
        return 0
    if src.count(OLD) != 1:
        raise SystemExit(f"anchor '{OLD}' found {src.count(OLD)} times in {path}")
    open(path, "w").write(src.replace(OLD, NEW, 1))
    print("FLA shmem patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
