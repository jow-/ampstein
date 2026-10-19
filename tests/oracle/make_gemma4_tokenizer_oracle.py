#!/usr/bin/env python3
"""Oracle fixture: Gemma 4 tokenizer (SentencePiece-style BPE with byte fallback) from the HF `tokenizers` reference.

    fixtures/gemma4_tokenizer_oracle.json : [ {"text": ..., "ids": [...]}, ... ]

Run from the repo root:
    python3 tests/oracle/make_gemma4_tokenizer_oracle.py [model_dir]
"""
import json
import os
import sys

from tokenizers import Tokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/models/gemma-4-26B-A4B-it-exl3-6.10bpw"
OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures", "gemma4_tokenizer_oracle.json")

tests = [
    "Hello, world!",
    "  leading and trailing spaces  ",
    "def foo(x):\n    return x * 2  # double\n\n\nprint(foo(21))",
    "The quick brown fox jumps over the lazy dog. 1234567890 3.14159",
    "Ünïcödé → 日本語のテキスト 🙂🚀 and emoji",
    "tabs\tand\n\nnewlines\n\n\n\nmore",
    "<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n",
    "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
    "a" * 300,
    "x     y        z",
    "\u0000\u0001 weird ퟿ chars   nbsp",
    "<|tool_call>call:foo{bar:<|\"|>baz<|\"|>}<tool_call|>",
    "word" * 50 + " end of the line, with punctuation!?;:'\"()[]{}",
]

t = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
out = [{"text": s, "ids": t.encode(s, add_special_tokens=False).ids} for s in tests]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"))
print(f"wrote {len(out)} cases to {OUT}")
