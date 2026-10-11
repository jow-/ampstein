#!/usr/bin/env python3
"""Oracle fixture: EXL3-quantized tensors + reference outputs (test-time only).

Quantizes deterministic fp16 matrices with exllamav3's own quantizer
(uncalibrated fallback path — meta Hessian) and saves, per bitrate:

    fixtures/exl3/K<k>.safetensors :
        trellis   uint16 (k/16, n/16, 16*K)   packed quantized weight
        suh       float16 (k)                 input Hadamard scales
        svh       float16 (n)                 output Hadamard scales
        W_dequant float16 (k, n)              oracle reconstruct(W)
        W_orig    float16 (k, n)              oracle reconstruct_had_slice
                                                (Hadamard + scales applied)
        X         float16 (m, k)              deterministic input
        Y_raw     float16 (m, n)              oracle X @ W_dequant
        Y_orig    float16 (m, n)              oracle X @ W_orig

    Orientation matters: exl3_gemm computes C = A @ dequant(B) with the
    dequantized weight as a (k, n) matrix — NO transpose (see upstream
    tests/test_sm75_gemm.py, whose reference is `A @ B`). The first draft
    of this script stored X @ W.T, which silently compares against the
    transposed weight and fails against the kernel by a huge margin.

Run from repo root:  .venv/bin/python tests/oracle/make_exl3_fixture.py
"""
import os
import torch
from safetensors.torch import save_file
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

torch.manual_seed(0)
device = "cuda:0"

outdir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(outdir, exist_ok=True)

# (in_features, out_features) divisible by 16/128 respectively
k, n = 256, 256
m = 8

for K in (2, 4, 8):
    W = torch.linspace(-1.0, 1.0, k * n, dtype=torch.float).reshape(k, n)
    W += torch.linspace(0.05, 0.5, k * n, dtype=torch.float).reshape(n, k).T
    W = W.to(device)

    H_data = {
        # meta tensor must carry the real shape: finalize_capture_H derives
        # the input-channel count from H.shape[0] in the fallback path
        "H": torch.empty((k, k), dtype=torch.float, device="meta"),
        "device": device,
    }
    quant_args = {
        "K": K,
        "seed": 0,
        "sigma_reg": 0.025,
        "devices": [device],
        "device_ratios": [1.0],
        "apply_out_scales": None,
        "debug_dir": "/tmp/exl3_debug",
    }

    result = quantize_exl3(W, H_data, quant_args, True, None, False)
    # result: (quantized weight, proxy error, packed tensors)
    _, err, packed = result

    trellis = packed["trellis"].cpu()
    suh = packed["suh"].flatten().to(torch.float16).cpu()
    svh = packed["svh"].flatten().to(torch.float16).cpu()

    # oracle dequantized weights via the native ext:
    #   W_dequant — raw tile dequant (Hadamard basis, no scales):
    #       matches exl3_gemm WITHOUT suh/svh
    #   W_orig    — original-basis weight (Hadamard + scales applied):
    #       matches exl3_gemm WITH suh/svh, and reconstruct_had_slice
    W_dq = torch.empty(k, n, dtype=torch.float16, device=device)
    ext.reconstruct(W_dq, trellis.to(device), float(K), False, False)

    W_orig = torch.empty(k, n, dtype=torch.float16, device=device)
    ext.reconstruct_had_slice(W_orig, trellis.to(device),
                              suh.to(device), svh.to(device),
                              float(K), False, False, 0)
    torch.cuda.synchronize()

    X = torch.linspace(-0.5, 0.5, m * k, dtype=torch.float16).reshape(m, k).to(device)
    # No transpose: exl3_gemm consumes the dequantized (k, n) weight directly
    # (upstream test_sm75_gemm.py uses the same orientation for its reference).
    Y_raw = X @ W_dq      # reference for gemm without scales
    Y_orig = X @ W_orig   # reference for gemm with suh/svh

    save_file({
        "trellis": trellis,
        "suh": suh,
        "svh": svh,
        "W_dequant": W_dq.cpu(),
        "W_orig": W_orig.cpu(),
        "X": X.cpu(),
        "Y_raw": Y_raw.cpu(),
        "Y_orig": Y_orig.cpu(),
    }, os.path.join(outdir, f"exl3_K{K}.safetensors"))

    print(f"K={K}: trellis {tuple(trellis.shape)} err={err}")

print("done")