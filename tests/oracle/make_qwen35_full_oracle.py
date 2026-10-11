#!/usr/bin/env python3
"""Oracle fixture: full-model logits + greedy continuation for the
Qwen3.8-27B exl3 model (exllamav3, autosplit over the available GPUs).

    fixtures/qwen35_full.safetensors :
        ids      int64  (S,)     prompt token ids (BOS first)
        logits   f16    (S, V)   final logits for every prompt position
        cont     int64  (T,)     greedy continuation ids (excluding the
                                 prompt's last token), T = CONT_TOKENS

Run from repo root (CUDA needed, takes a few minutes to load):
    .venv/bin/python tests/oracle/make_qwen35_full_oracle.py
"""
import os

import torch
from safetensors.torch import save_file
from exllamav3 import Model, Config, Tokenizer

MODEL_DIR = "/data/models/Qwen3.8-27B-exl3-5.00bpw"
PROMPT = "The capital of France is Paris. The capital of Germany is"
CONT_TOKENS = 8
device = "cuda:0"
torch.cuda.set_device(0)

outdir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(outdir, exist_ok=True)

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
tok = Tokenizer(config)

print("loading (autosplit)...", flush=True)
model.load()

ids = tok.encode(PROMPT, add_bos=True).reshape(-1).tolist()
print("ids:", ids, flush=True)

with torch.inference_mode():
    logits = model.forward(torch.tensor([ids], dtype = torch.long, device = device))
    print("logits", tuple(logits.shape), flush=True)

    # greedy continuation (stateless forwards over the growing prefix)
    cont = []
    work = list(ids)
    for i in range(CONT_TOKENS):
        lg = model.forward(torch.tensor([work], dtype = torch.long, device = device))
        nxt = int(lg[0, -1, :].argmax())
        cont.append(nxt)
        work.append(nxt)
        print("token", i, "->", nxt, flush=True)

save_file({
    "ids": torch.tensor(ids, dtype=torch.int64),
    "logits": logits[0].half().cpu(),
    "cont": torch.tensor(cont, dtype=torch.int64),
}, os.path.join(outdir, "qwen35_full.safetensors"))
print("saved fixture.", flush=True)

model.unload()
print("done.", flush=True)