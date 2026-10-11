#!/usr/bin/env python3
"""Oracle test fixture generator (test-time only, never shipped).

Creates tests/fixtures/st_test.safetensors with deterministic values so
ucode-level tests can compare bit-exactly. Run from the repo root:

    .venv/bin/python tests/oracle/make_fixture.py
"""
import os
import torch
from safetensors.torch import save_file

tensors = {
    # deterministic fp16 matrix (linspace guarantees exact fp16 representation)
    "w": torch.linspace(-2.0, 2.0, 32, dtype=torch.float16).reshape(4, 8),
    "b": torch.arange(8, dtype=torch.int32),
    "flag": torch.tensor([1, 0, 1], dtype=torch.int64),
}

out = os.path.join(os.path.dirname(__file__), "..", "fixtures", "st_test.safetensors")
os.makedirs(os.path.dirname(out), exist_ok=True)
save_file(tensors, out)
print("wrote", os.path.abspath(out))
print("oracle w row0:", tensors["w"][0].tolist())