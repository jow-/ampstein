#!/usr/bin/env python3
"""Oracle fixture: tokenizer encode/decode pairs from a REAL model dir.

Runs a battery of strings through exllamav3's Tokenizer (which wraps the
HF `tokenizers` reference implementation) and saves text -> ids pairs:

    fixtures/tokenizer_oracle.json :
        { "cases": [ { "text": ..., "ids": [...] }, ... ] }

The battery covers ASCII, contractions, digit runs, UTF-8 (latin,
emoji, CJK), whitespace runs, special tokens (standalone and embedded),
and repeated text. This is the Phase 3 oracle for tests/tokenizer.uc:
our native tokenizer must reproduce these ids exactly.

Run from repo root (needs MODEL_DIR below):
    .venv/bin/python tests/oracle/make_tokenizer_oracle.py
"""
import json
import os

import torch
from exllamav3.model.config import Config
from exllamav3.tokenizer import Tokenizer

MODEL_DIR = "/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw"

CASES = [
    "The capital of France is",
    "Hello, world!",
    "",
    " ",
    "  leading spaces\nand\ttabs 12345",
    "trailing   ",
    "café über naïve",
    "emoji 😀 test 🇪🇺",
    "日本語のテキスト",
    "it's I'll we've they're I'm you'd don't",
    "12345 678 9",
    "function(x) { return x + 1; }",
    "a" * 50,
    "mixed123 numbers456 and symbols?!",
    "line1\nline2\n\nline3\r\nline4",
    "<|begin_of_text|>Hello",
    "a<|end_of_text|>b",
    "<|begin_of_text|>a<|end_of_text|><|finetune_right_pad_id|>",
    "code: if (x == 42) { y = \"str\"; }",
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs.",
    "3.14159265358979 2.71828 1.618",
    "e-mail: someone@example.com http://x.y/z?a=1&b=2",
    "Ω≈ç√∫˜µ≤≥÷",
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 astral 𝕏 plane",
    "tab\tseparated\tvalues",
    "!!???...,,,;;;:::",
    "CamelCase snake_case kebab-case",
    "<|start_header_id|>assistant<|end_header_id|>",
    # ignore_merges discriminators: vocab entries the merge path alone
    # cannot reach (llama-3 sets model.ignore_merges = true; the engine
    # must emit the whole pre-token from the vocab without merging)
    "ektedir", "mektedir", "lardan", "iyordu", "ecektir",
    ".:.:.:.:.:.:.:.:",
    "the ektedir and iyordu mixed with ordinary English words",
]

outdir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(outdir, exist_ok=True)

config = Config.from_directory(MODEL_DIR)
tok = Tokenizer(config)

cases = []

for text in CASES:
    ids = tok.encode(text, add_bos=False)
    ids = ids.reshape(-1).tolist()
    case = {"text": text, "ids": ids}

    # special-token strings encode atomically only with the flag set;
    # record both variants when they differ
    ids_sp = tok.encode(text, add_bos=False, encode_special_tokens=True)
    ids_sp = ids_sp.reshape(-1).tolist()

    if ids_sp != ids:
        case["ids_special"] = ids_sp

    cases.append(case)

    # sanity: the reference must roundtrip its own encoding
    back = tok.decode(torch.tensor([ids]), decode_special_tokens=True)

    if back[0] != text:
        raise RuntimeError(f"reference does not roundtrip: {text!r} -> {back!r}")

with open(os.path.join(outdir, "tokenizer_oracle.json"), "w") as f:
    json.dump({"cases": cases}, f, ensure_ascii=False, indent=1)

print(f"wrote {len(cases)} cases")

# --- pair-form merges fixture -------------------------------------------
#
# Newer `tokenizers` releases write model.merges as [["a", "b"], ...]
# pairs instead of "a b" strings. Build a tokenizer.json with the pair
# encoding (same vocab, same merges, same regex) and record HF ids for
# a few strings: the native engine must load it and produce the same
# ids as the reference does on the SAME file.
import shutil

from tokenizers import Tokenizer as HFTokenizer

pair_dir = os.path.join(outdir, "tokenizer_pairform")
os.makedirs(pair_dir, exist_ok=True)

with open(os.path.join(MODEL_DIR, "tokenizer.json")) as f:
    tj = json.load(f)

merges = tj["model"]["merges"]
assert isinstance(merges[0], str), "reference model uses pair form already"
tj["model"]["merges"] = [m.split(" ", 1) for m in merges]

with open(os.path.join(pair_dir, "tokenizer.json"), "w") as f:
    json.dump(tj, f, ensure_ascii=False)

hf = HFTokenizer.from_file(os.path.join(pair_dir, "tokenizer.json"))

pair_cases = []
for text in ["The capital of France is", "Hello, world!", "it's I'll we've",
             "12345 678 9", "café über naïve", "ektedir"]:
    enc = hf.encode(text, add_special_tokens=False)
    pair_cases.append({"text": text, "ids": enc.ids})

with open(os.path.join(outdir, "tokenizer_pairform.json"), "w") as f:
    json.dump({"dir": pair_dir, "cases": pair_cases}, f, ensure_ascii=False,
              indent=1)

print(f"wrote pair-form fixture ({len(pair_cases)} cases) -> {pair_dir}")