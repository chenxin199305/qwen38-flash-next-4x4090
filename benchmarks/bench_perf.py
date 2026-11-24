#!/usr/bin/env python3
"""Prefill and decode benchmark for an OpenAI-compatible vLLM endpoint.

Prefill: unique random prompt at each target size, TTFT measured to the first
content/reasoning token. Prefix caching cannot help because every request uses
fresh text. Decode: fixed max_tokens with ignore_eos, token counts from
usage.completion_tokens (speculative decoding emits several tokens per chunk).

Examples:
    python benchmarks/bench_perf.py --api-key-file ~/.vllm-key
    python benchmarks/bench_perf.py --prefill-sizes 4096,16384,65536,131072 \
        --decode-runs 3 --max-tokens 256 --output benchmarks/results/local.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import string
import time
import urllib.request


def build_headers(args) -> dict:
    key = args.api_key or os.environ.get("VLLM_API_KEY", "")
    if args.api_key_file and not key:
        key = open(args.api_key_file).readline().strip()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def post(args, path, payload, timeout=1800):
    req = urllib.request.Request(
        args.base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers=build_headers(args),
    )
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(args, text: str) -> int:
    root = args.base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    req = urllib.request.Request(
        root + "/tokenize",
        data=json.dumps({"model": args.model, "prompt": text}).encode(),
        headers=build_headers(args),
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["count"]


def make_text(args, target_tokens: int, seed: int):
    rng = random.Random(seed)
    words = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9)))
             for _ in range(int(target_tokens * 0.75))]
    text = " ".join(words)
    n = tokenize(args, text)
    factor = target_tokens / max(n, 1)
    words = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9)))
             for _ in range(max(int(len(words) * factor), 16))]
    text = " ".join(words)
    return text, tokenize(args, text)


def stream_chat(args, content: str, max_tokens: int, ignore_eos: bool):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    t0 = time.time()
    first = last = None
    chunks = 0
    usage = None
    with post(args, "/chat/completions", payload) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or delta.get("reasoning_content")
                         or delta.get("reasoning"))
                if piece:
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
                    chunks += 1
    return t0, first, last, chunks, usage


def decode_bench(args):
    rows = []
    for i in range(args.decode_runs):
        prompt = f"请写一段约300字关于人工智能的短文。批次{i}-{time.time_ns()}"
        t0, first, last, _, usage = stream_chat(
            args, prompt, args.max_tokens, args.ignore_eos)
        ct = (usage or {}).get("completion_tokens", 0)
        pt = (usage or {}).get("prompt_tokens", 0)
        rate = (ct - 1) / (last - first) if first and last and ct > 1 else 0
        rows.append({
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "ttft_s": round(first - t0, 4) if first else None,
            "decode_tok_s": round(rate, 1),
        })
        print(f"[decode {i+1}] prompt={pt} completion={ct} "
              f"TTFT={rows[-1]['ttft_s']}s decode={rows[-1]['decode_tok_s']} tok/s",
              flush=True)
    return rows


def prefill_bench(args):
    rows = []
    for target in args.prefill_sizes:
        warm, _ = make_text(args, target, time.time_ns())
        stream_chat(args, warm + "\n\n只回复 OK。", 1, False)
        text, ntok = make_text(args, target, time.time_ns() + 1)
        t0, first, _, _, usage = stream_chat(
            args, text + "\n\n只回复 OK。", 1, False)
        pt = (usage or {}).get("prompt_tokens", ntok)
        ttft = first - t0 if first else float("nan")
        rate = pt / ttft if ttft else 0
        rows.append({
            "target_tokens": target,
            "prompt_tokens": pt,
            "ttft_s": round(ttft, 3),
            "prefill_tok_s": round(rate, 1),
        })
        print(f"[prefill] target={target} prompt={pt} "
              f"TTFT={rows[-1]['ttft_s']}s rate={rows[-1]['prefill_tok_s']} tok/s",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:12345/v1")
    ap.add_argument("--model", default="qwen38-flash-next-nvfp4")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--decode-runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ignore-eos", action="store_true", default=True)
    ap.add_argument("--prefill-sizes", default="4096,16384,65536,131072")
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    args.prefill_sizes = [int(x) for x in args.prefill_sizes.split(",")]

    stream_chat(args, "hello", 8, False)
    print("== decode ==", flush=True)
    decode = decode_bench(args)
    print("== prefill ==", flush=True)
    prefill = prefill_bench(args)

    result = {
        "base_url": args.base_url,
        "model": args.model,
        "decode": decode,
        "decode_median_tok_s": round(statistics.median(
            r["decode_tok_s"] for r in decode), 1) if decode else 0,
        "prefill": prefill,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
