#!/usr/bin/env python3
"""Oracle fixture: one EXL3 layer from a REAL model dir (test-time only).

Extracts the q_proj of layer 0 from a real exllamav3 model directory and
computes the exllamav3 reference outputs, saving:

    fixtures/exl3_real_layer.safetensors :
        trellis   int16   (k/16, n/16, 16*K)   packed quantized weight
        suh       float16 (k)                  input Hadamard scales
        svh       float16 (n)                  output Hadamard scales
        W_dequant float16 (k, n)                oracle reconstruct()
                                                (Hadamard basis, no scales)
        W_orig    float16 (k, n)                oracle reconstruct_had_slice()
                                                (original basis: Hadamard +
                                                scales applied)
        X         float16 (m, k)                deterministic input
        Y         float16 (m, n)                oracle X @ W_orig
                                                (NO transpose — see
                                                make_exl3_fixture.py / L15;
                                                gemm with suh/svh matches
                                                this reference)

Source model (not redistributed): turboderp/Llama-3.2-1B-Instruct-exl3,
branch 4.0bpw. Downloaded to MODEL_DIR below (config.json +
model.safetensors). The fixture carries only the tensors of one layer, so
it stays small; it is derived from real published weights and is
deterministic given the model.

Run from repo root:  .venv/bin/python tests/oracle/make_exl3_real_fixture.py
"""
import json
import os
import struct

import torch
from safetensors.torch import load_file, save_file
from exllamav3.ext import exllamav3_ext as ext

MODEL_DIR = "/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw"
LAYER = "model.layers.0.self_attn.q_proj"
M = 8

device = "cuda:0"
torch.cuda.set_device(0)

outdir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(outdir, exist_ok=True)

st = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
trellis = st[f"{LAYER}.trellis"].to(device)
suh = st[f"{LAYER}.suh"].to(device)
svh = st[f"{LAYER}.svh"].to(device)

k = trellis.shape[0] * 16
n = trellis.shape[1] * 16
K = trellis.shape[2] / 16
assert K.is_integer(), "fixture layer must be an integer bitrate"
K = int(K)

# oracle dequantized weights, both bases:
#   W_dequant — plain reconstruct(): Hadamard basis, scales NOT applied
#   W_orig    — reconstruct_had_slice(): original basis (Hadamard + scales)
# exl3_gemm with suh/svh consumes the W_orig basis (no transpose, L15)
W_dq = torch.empty(k, n, dtype=torch.float16, device=device)
ext.reconstruct(W_dq, trellis, float(K), False, False)

W = torch.empty(k, n, dtype=torch.float16, device=device)
ext.reconstruct_had_slice(W, trellis, suh, svh, float(K), False, False, 0)

X = torch.linspace(-0.5, 0.5, M * k, dtype=torch.float16).reshape(M, k).to(device)
Y = X @ W  # no transpose (L15)

save_file({
    "trellis": trellis.cpu(),
    "suh": suh.cpu(),
    "svh": svh.cpu(),
    "W_dequant": W_dq.cpu(),
    "W_orig": W.cpu(),
    "X": X.cpu(),
    "Y": Y.cpu(),
}, os.path.join(outdir, "exl3_real_layer.safetensors"))

print(f"{LAYER}: k={k} n={n} K={K}, source {MODEL_DIR}")
print("done")