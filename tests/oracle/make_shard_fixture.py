#!/usr/bin/env python3
"""Oracle fixture: a SHARDED model directory for the weight-map tests.

Takes the existing exl3_real_layer.safetensors fixture (real weights of
layer-0 q_proj of Llama-3.2-1B-exl3, see make_exl3_real_fixture.py) and
repackages it as an HF-style sharded model dir:

    fixtures/sharded/
        model.safetensors.index.json
        model-00001-of-00002.safetensors   trellis, suh, svh
        model-00002-of-00002.safetensors   W_dequant, W_orig, X, Y
                                           + a plain fp16 linear:
                                           proj.weight, proj.bias,
                                           proj_x, proj_y (reference)

The EXL3 keys and their gemm references come from the single-file
fixture unchanged, so tests/weights.uc can compare the routed tensors
bit-exactly against the un-sharded originals. The fp16 linear carries
its own torch-computed reference Y = X @ W.T + bias.

Run from repo root:
    .venv/bin/python tests/oracle/make_shard_fixture.py
"""
import json
import os

import torch
from safetensors.torch import load_file, save_file

SRC = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                   "exl3_real_layer.safetensors")
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sharded")

os.makedirs(OUTDIR, exist_ok=True)

tensors = load_file(SRC)

shard1 = {"q_proj." + k: tensors[k] for k in ("trellis", "suh", "svh")}
shard2 = {k: tensors[k] for k in ("W_dequant", "W_orig", "X", "Y")}

# a plain fp16 linear with bias, plus its torch-computed reference
# (computed on CUDA, the device our ucode matmul runs on)
torch.manual_seed(0)
W = torch.randn(64, 32, dtype=torch.float16)
bias = torch.randn(64, dtype=torch.float16)
X = torch.randn(4, 32, dtype=torch.float16)
Y = (X.cuda() @ W.cuda().T + bias.cuda()).cpu()

shard2["proj.weight"] = W.contiguous()
shard2["proj.bias"] = bias.contiguous()
shard2["proj_x"] = X.contiguous()
shard2["proj_y"] = Y.contiguous()

# a bf16 tensor to exercise the fp16 cast path
shard2["norm.weight"] = torch.randn(32, dtype=torch.bfloat16).contiguous()

save_file(shard1, os.path.join(OUTDIR, "model-00001-of-00002.safetensors"))
save_file(shard2, os.path.join(OUTDIR, "model-00002-of-00002.safetensors"))

weight_map = {}
for name, shard in (("model-00001-of-00002.safetensors", shard1),
                    ("model-00002-of-00002.safetensors", shard2)):
    for k in shard:
        weight_map[k] = name

index = {
    "metadata": {"total_size": 0},
    "weight_map": weight_map,
}

with open(os.path.join(OUTDIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

print("wrote", OUTDIR)
print("keys shard1:", sorted(shard1))
print("keys shard2:", sorted(shard2))