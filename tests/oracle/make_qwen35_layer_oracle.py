#!/usr/bin/env python3
"""Oracle fixture: per-layer hidden states for the Qwen3.8-27B exl3
hybrid architecture, captured through exllamav3's own module chain.

Loads ONLY the embed, the first four blocks (three GDN linear-attention
layers + one full-attention layer), the final norm and lm_head onto one
GPU (~6 GB) and walks them module by module — the eval/model_diff.py
harness shape. The ucode port (tests/qwen35_layers.uc) replays the same
chain and compares every boundary.

    fixtures/qwen35_layers.safetensors :
        ids        int64   (S,)       input token ids (BOS first)
        h_embed    f16     (S, hid)   embedding output
        h_block0   f16     (S, hid)   after GDN block 0
        h_block1   f16     (S, hid)   after GDN block 1
        h_block2   f16     (S, hid)   after GDN block 2
        h_block3   f16     (S, hid)   after full-attention block 3
        logits     f16     (S, V)     final logits (norm + lm_head)

Run from repo root (CUDA needed):
    .venv/bin/python tests/oracle/make_qwen35_layer_oracle.py
"""
import os

import torch
from safetensors.torch import save_file
from exllamav3 import Model, Config
from exllamav3.tokenizer import Tokenizer

MODEL_DIR = "/data/models/Qwen3.8-27B-exl3-5.00bpw"
PROMPT = "The capital of France is Paris. The capital of"
device = "cuda:0"
torch.cuda.set_device(0)

outdir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(outdir, exist_ok=True)

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
tok = Tokenizer(config)

ids_t = tok.encode(PROMPT, add_bos=True)
ids = ids_t.reshape(-1).tolist()
print("ids:", ids)

# The subset of modules we need, in chain order.
wanted_keys = [
    "model.language_model.embed_tokens",
] + [f"model.language_model.layers.{i}" for i in range(4)] + [
    "model.language_model.norm",
    "lm_head",
]

modules = []
config.stc.begin_deferred_load()
for module in model.modules:
    if getattr(module, "key", None) in wanted_keys:
        module.load(device)
        modules.append(module)
config.stc.end_deferred_load()

print("loaded modules:", [m.key for m in modules])
assert len(modules) == len(wanted_keys)

input_ids = torch.tensor([ids], dtype=torch.long, device=device)

outs = {}
x = input_ids
params = {}
for module in modules:
    x = module.forward(x, params)
    name = module.key.split(".")[-1] + (
        "_" + module.key.split(".")[-2] if "layers" in module.key else "")
    if module.key.startswith("model.language_model.layers"):
        name = "h_block" + module.key.split(".")[-1]
    elif module.key.endswith("embed_tokens"):
        name = "h_embed"
    elif module.key.endswith(".norm"):
        name = "h_norm"
    elif module.key == "lm_head":
        name = "logits"
    if name == "h_norm":
        continue  # norm output folded into logits comparison
    outs[name] = x[0].half().cpu()
    print(name, tuple(x.shape), x.dtype, flush=True)

outs["ids"] = torch.tensor(ids, dtype=torch.int64)
save_file(outs, os.path.join(outdir, "qwen35_layers.safetensors"))
print("saved fixture.", flush=True)