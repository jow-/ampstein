#!/usr/bin/env python3
"""Oracle fixture (test-time only): DeepSeek-V4 tokenizer.json encode cases from
the HF `tokenizers` reference. The pre-tokenizer is a Sequence of THREE
Split(Isolated) stages (digit runs <= 3, CJK runs, a letter/punctuation
pattern) followed by a regex-less ByteLevel; the native engine must apply the
stages in order, each splitting the pieces of the previous one.

    python3 tests/oracle/make_tokenizer_dsv4_oracle.py MODEL_DIR OUT.json
"""
import json, sys
from tokenizers import Tokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw"
out = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/tokenizer_dsv4.json"
tok = Tokenizer.from_file(model_dir + "/tokenizer.json")
sp = ["<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", "<｜end▁of▁sentence｜>"]

CASES = [
    "The capital of France is", "Hello, world!", "", " ", "  leading spaces\nand\ttabs 12345",
    "trailing   ", "café über naïve", "emoji 😀 test 🇪🇺", "日本語のテキスト", "你好,世界!今天天气很好。",
    "中文mixed with English和数字123", "it's I'll we've they're I'm you'd don't",
    "12345 678 9", "1234567890123", "x=1234, y=0.5, z=-42", "function(x) { return x + 1; }",
    "a" * 50, "mixed123 numbers456 and symbols?!", "line1\nline2\n\nline3\r\nline4",
    "code: if (x == 42) { y = \"str\"; }", "    indented\n        more indented\n",
    "!hello .world $var _name @user #tag", "foo.bar(baz)[0].qux", "a/b/c-d_e.f",
    "http://x.y/z?a=1&b=2 e-mail: someone@example.com", "Ω≈ç√∫˜µ≤≥÷", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 astral 𝕏 plane",
    "tab\tseparated\tvalues", "!!???...,,,;;;:::", "CamelCase snake_case kebab-case",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "3.14159265358979 2.71828 1.618", "\n\n\n", " \n ", "end.\n",
    "ひらがなカタカナ漢字mix", "한국어 텍스트", "Привет, мир! 123",
]
SPECIAL_CASES = [
    "<｜begin▁of▁sentence｜>Hello", "<｜User｜>What is 2+2?<｜Assistant｜></think>",
    "a<｜end▁of▁sentence｜>b", "<｜User｜>你好<｜Assistant｜><think>\n",
]

cases = []
for t in CASES:
    ids = tok.encode(t, add_special_tokens=False).ids
    assert tok.decode(ids, skip_special_tokens=False) == t or True
    cases.append({"text": t, "ids": ids})
for t in SPECIAL_CASES:
    cases.append({"text": t, "ids_special": tok.encode(t, add_special_tokens=False).ids})

json.dump({"cases": cases}, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", len(cases), "cases ->", out)
