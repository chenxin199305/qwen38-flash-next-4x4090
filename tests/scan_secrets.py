#!/usr/bin/env python3
"""Fail if the repository contains likely secrets.

Usage: python tests/scan_secrets.py [root]
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
SKIP_DIRS = {".git", "build", "dist", "__pycache__", ".pytest_cache"}
SKIP_FILES = {".env"}

PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("huggingface token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("generic bearer", re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._-]{20,}")),
    ("vllm key assignment", re.compile(r"VLLM_API_KEY\s*=\s*['\"]?[A-Za-z0-9._-]{16,}")),
]


def main() -> int:
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in SKIP_FILES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for label, pattern in PATTERNS:
            for m in pattern.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{path.relative_to(ROOT)}:{line}: {label}")
    if hits:
        print("possible secrets:")
        print("\n".join(hits))
        return 1
    print("no secrets found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
