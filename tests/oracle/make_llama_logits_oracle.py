#!/usr/bin/env python3
"""Oracle fixture: full-model prefill logits from a REAL model dir.

Runs turboderp/Llama-3.2-1B-Instruct-exl3 (4.0bpw branch) through
exllamav3's own Model.forward — the complete reference graph: embedding,
RMSNorm, RoPE (incl. llama3 scaling), GQA attention, SwiGLU MLP, quantized
lm_head — and saves the logits for a fixed token sequence:

    fixtures/llama_logits.safetensors :
        ids       int64   (S,)     input token ids
        logits    float32 (S, V)   oracle logits for every position

This is the Phase 3 oracle for tests/llama_forward.uc: our ucode forward
graph must reproduce these logits (fp16 noise tolerance, argmax match).

Run from repo root (needs MODEL_DIR below + CUDA):
    .venv/bin/python tests/oracle/make_llama_logits_oracle.py
"""
import os

import torch
from safetensors.torch import save_file
from exllamav3 import Model, Config
from exllamav3.tokenizer import Tokenizer

MODEL_DIR = "/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw"
PROMPT = "The capital of France is"

device = "cuda:0"
torch.cuda.set_device(0)

outdir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(outdir, exist_ok=True)

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
model.load()

# Tokenize with the model's own tokenizer so the ids are real text ids.
tok = Tokenizer(config)
ids_t = tok.encode(PROMPT)
ids = ids_t.reshape(-1).tolist()
print("token ids:", ids)

input_ids = torch.tensor([ids], dtype=torch.long, device=device)
with torch.inference_mode():
    logits = model.forward(input_ids)

print("forward output shape:", tuple(logits.shape))
if logits.dim() == 3:
    logits = logits[0]
elif logits.dim() == 2 and logits.shape[0] == 1 and logits.shape[-1] != model.config.vocab_size:
    logits = logits[0]

logits = logits.float().cpu()
print("logits:", tuple(logits.shape))

argmax = [int(logits[i].argmax()) for i in range(logits.shape[0])]
print("per-position argmax:", argmax)

save_file(
    {
        "ids": torch.tensor(ids, dtype=torch.int64),
        "logits": logits,
    },
    os.path.join(outdir, "llama_logits.safetensors"),
)
print("wrote", os.path.join(outdir, "llama_logits.safetensors"))