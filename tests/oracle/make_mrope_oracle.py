#!/usr/bin/env python3
"""Oracle fixture (test-time only): MRoPE cos/sin rows for a prompt with
two images, from exllamav3's own RoPE.get_mrope_freqs (CPU, no weights).

    tests/fixtures/mrope_oracle.safetensors :
        ids    i64 (S,)      token ids (image rows carry exllamav3's
                             allocated embedding ids)
        cos    f32 (S, 32)   cos of the interleaved MRoPE angles
        sin    f32 (S, 32)
        meta   i64 (6,)      [img1_start, img1_n, img2_start, img2_n,
                              next_pos, S]

Run: .venv/bin/python tests/oracle/make_mrope_oracle.py [MODEL_DIR]
"""
import os, sys
from types import SimpleNamespace
import torch
from safetensors.torch import save_file
from exllamav3 import Config
from exllamav3.util.rope import RoPE

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/models/Qwen3.8-27B-exl3-5.00bpw"
outdir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(outdir, exist_ok = True)

config = Config.from_directory(MODEL_DIR)
rope = RoPE("cpu", config.rope_settings)

grids = [(1, 14, 20), (1, 8, 6)]
ns = [g[1] // 2 * (g[2] // 2) for g in grids]
first = [10_000_000, 20_000_000]

ids = list(range(100, 105)) + [248053]
s1 = len(ids); ids += [first[0] + k for k in range(ns[0])]
ids += [248054] + list(range(200, 207)) + [248053]
s2 = len(ids); ids += [first[1] + k for k in range(ns[1])]
ids += [248054] + list(range(300, 304))

embs = [
    SimpleNamespace(first_index = first[i], last_index = first[i] + ns[i],
                    grid_thw = grids[i], mrope_merge_size = 2)
    for i in range(2)
]
t = torch.tensor([ids], dtype = torch.long)
freqs, nxt = rope.get_mrope_freqs(t, embs, t.shape[-1])
freqs = freqs[0]
print("freqs", tuple(freqs.shape), "next", nxt, "S", len(ids))

save_file({
    "ids": torch.tensor(ids, dtype = torch.int64),
    "cos": torch.cos(freqs).float().contiguous(),
    "sin": torch.sin(freqs).float().contiguous(),
    "meta": torch.tensor([s1, ns[0], s2, ns[1], nxt, len(ids)], dtype = torch.int64),
}, os.path.join(outdir, "mrope_oracle.safetensors"))
