#!/usr/bin/env python3
"""Oracle fixture: full 16-layer EXL3 linear stack from a REAL model dir.

Runs every layer's 7 EXL3 linears (q/k/v/o + gate/up/down) of
turboderp/Llama-3.2-1B-Instruct-exl3 (4.0bpw branch) through exllamav3's
own LinearEXL3.forward, with a deterministic input, and saves the
per-layer down_proj outputs:

    fixtures/exl3_stack_oracle.safetensors :
        X          float16 (2, 2048)   deterministic input
        layer{i}_Y float16 (2, 2048)   oracle output of layer i's stack:
                                       q = q_proj(X); attn = o_proj(q)
                                       gate = gate_proj(attn)
                                       up = up_proj(attn)
                                       act = silu(gate) * up
                                       Y = down_proj(act)

The dataflow is deliberately the same as ucode/models.uc's test: each
layer is exercised independently (no cross-layer hidden state), so the
fixture isolates the linear stack.

Run from repo root (needs MODEL_DIR below):
    .venv/bin/python tests/oracle/make_exl3_stack_oracle.py
"""
import os

import torch
from safetensors.torch import load_file, save_file
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3

MODEL_DIR = "/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw"
NUM_LAYERS = 16
M = 2

device = "cuda:0"
torch.cuda.set_device(0)

outdir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(outdir, exist_ok=True)

st = load_file(os.path.join(MODEL_DIR, "model.safetensors"))

def layer(i):
    out = {}
    for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        key = f"model.layers.{i}.{name}"
        s = st[f"{key}.trellis"].shape
        out[name] = LinearEXL3(config=None, in_features=s[0]*16, out_features=s[1]*16,
                               suh=st[f"{key}.suh"].to(device), svh=st[f"{key}.svh"].to(device),
                               trellis=st[f"{key}.trellis"].to(device))
    return out

layers = [layer(i) for i in range(NUM_LAYERS)]
x = torch.full((M, 2048), 0.1, dtype=torch.float16, device=device)

out_tensors = {"X": x.cpu()}
for i in range(NUM_LAYERS):
    L = layers[i]
    q = L["self_attn.q_proj"].forward(x, {})
    attn = L["self_attn.o_proj"].forward(q, {})
    gate = L["mlp.gate_proj"].forward(attn, {})
    up = L["mlp.up_proj"].forward(attn, {})
    act = torch.nn.functional.silu(gate) * up
    down = L["mlp.down_proj"].forward(act, {})
    out_tensors[f"layer{i}_Y"] = down.cpu()

save_file(out_tensors, os.path.join(outdir, "exl3_stack_oracle.safetensors"))
print("saved", len(out_tensors), "tensors")
