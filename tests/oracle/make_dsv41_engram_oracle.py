#!/usr/bin/env python3
"""Engram hash oracle for DeepSeek-V4.1-Flash (tests/dsv41_engram.uc): the table row ids DeepSeek's
inference/engram.py computes for a token sequence, to compare with ampstein's hashing.

  python3 make_dsv41_engram_oracle.py <ref_dir> <model_dir> <out.safetensors>

<ref_dir>: DeepSeek-V4.1-Flash/inference (engram.py), <model_dir>: the pack (config.json, tokenizer.json).
Needs torch (CPU), tokenizers, numpy, sympy, safetensors.
"""
import json
import sys
import types

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

ref_dir, model_dir, out = sys.argv[1:4]
sys.path.insert(0, ref_dir)
from engram import EngramLayout, NgramHashState  # noqa: E402

root = json.load(open(f"{model_dir}/config.json"))
tc = root.get("text_config", root)
args = types.SimpleNamespace(
    engram_layer_ids=tc["engram_layer_ids"], engram_max_ngram_size=tc["engram_max_ngram_size"],
    engram_n_heads=tc["engram_n_heads"], engram_vocab_size=tc["engram_vocab_size"],
    engram_num_embeddings=tc["engram_num_embeddings"], engram_head_dim=tc["engram_head_dim"],
    engram_pad_id=tc["engram_pad_token_id"], engram_compressed_vocab_size=tc["engram_compressed_vocab_size"],
    max_batch_size=1, max_seq_len=4096,
)


class Tok:
    def __init__(self, path):
        self.backend_tokenizer = Tokenizer.from_file(path)

    def __len__(self):
        return tc["vocab_size"]


layout = EngramLayout.from_args(args)
st = NgramHashState(args, layout, Tok(f"{model_dir}/tokenizer.json"))
g = torch.Generator().manual_seed(7)
T = 300
ids = torch.randint(3, tc["vocab_size"], (1, T), generator=g)
ids[0, 40] = 2          # a pad token in the middle: ampstein treats it like the start of the sequence
h = st(ids, 0)           # [1, T, layers, 24]
save_file({"ids": ids[0].contiguous(), "hash": h[0].contiguous()}, out)
print("wrote", out, tuple(h.shape))
