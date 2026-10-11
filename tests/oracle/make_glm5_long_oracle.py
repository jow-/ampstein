#!/usr/bin/env python3
"""Oracle fixture (test-time only): GLM-5.3-Flash on a prompt LONGER than
index_topk (2048), where the k-pool DSA indexer actually selects.

The prompt is ~3.5k tokens of numbered records followed by a retrieval
question, so a wrong selection shows up in the answer. Recorded:

    ids         i64 (T,)       prompt ids
    logits      f16 (16, V)    logits of the last 16 positions
    cont        i64 (N,)       greedy continuation (cache-less forwards)
    l3_attn_in  f16 (T, hid)   layer 3 attention input
    l3_attn_out f32 (T, hid)   layer 3 attention output
    l3_idx      i32 (64, K)    the indexer's selected token indices for the
                               last 64 queries of layer 3 (-1 padded)

    python3 tests/oracle/make_glm5_long_oracle.py [MODEL_DIR] [OUT]
"""
import os, sys, hashlib
import torch
from safetensors.torch import save_file
from exllamav3 import Config, Model, Tokenizer
from exllamav3.modules.transformer import TransformerBlock

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/models/GLM-5.3-Flash-exl3-3.05bpw"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "glm5_long_oracle.safetensors")
CONT = 12
NREC = 190
TARGET = 17


def code(i):
    return hashlib.sha1(f"rec{i}".encode()).hexdigest()[:8]


lines = [f"Record {i}: the secret code for item {i} is {code(i)}." for i in range(NREC)]
PROMPT = ("[gMASK]<sop><|user|>\nBelow is a list of records.\n" + "\n".join(lines) +
          f"\nWhat is the secret code for item {TARGET}? Answer with the code only."
          "<|assistant|>\n</think>")

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
tok = Tokenizer.from_config(config)
ids = tok.encode(PROMPT, encode_special_tokens = True)
T = ids.shape[-1]
print("prompt tokens:", T, "expected answer:", code(TARGET), flush = True)
assert T > 2048 + 256, "prompt must exceed index_topk"

if os.environ.get("DRY"):
    sys.exit(0)

rec = {}
armed = {"on": False}
blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
b3 = blocks[3]
orig_fwd = b3.attn.forward


def attn_fwd(x, params, *a, **k):
    if armed["on"]:
        rec["l3_attn_in"] = x.detach().cpu()
    y = orig_fwd(x, params, *a, **k)
    if armed["on"]:
        rec["l3_attn_out"] = y.detach().cpu()
    return y


b3.attn.forward = attn_fwd
orig_sel = b3.attn._indexer_topk_kpool


def sel(*a, **k):
    out = orig_sel(*a, **k)
    if armed["on"]:
        rec["l3_idx"] = out[-64:].detach().cpu().to(torch.int32)
    return out


b3.attn._indexer_topk_kpool = sel

# residual streams after every block, last 64 rows only
for blk in blocks:
    def mk(i, blk):
        orig = blk.forward

        def f(x, params, *a, **k):
            y = orig(x, params, *a, **k)
            if armed["on"]:
                rec[f"x{i}"] = y[0, -64:].detach().cpu().float()
            return y
        blk.forward = f
    mk(blk.layer_idx, blk)

print("loading...", flush = True)
model.load(progressbar = True)

with torch.inference_mode():
    armed["on"] = True
    logits = model.forward(ids, params = {"attn_mode": "flash_attn_nc", "last_tokens_only": 16})
    armed["on"] = False
    print("logits", tuple(logits.shape), flush = True)
    rec["ids"] = ids.reshape(-1).to(torch.int64).cpu()
    rec["logits"] = logits[0].half().cpu()

    cont, work = [], ids.clone()
    for _ in range(CONT):
        lg = model.forward(work, params = {"attn_mode": "flash_attn_nc", "last_tokens_only": 1})
        nxt = int(lg[0, -1].argmax())
        cont.append(nxt)
        work = torch.cat([work, torch.tensor([[nxt]], dtype = work.dtype)], dim = 1)
    print("cont", cont, flush = True)
    print("answer text:", tok.decode(torch.tensor([cont])), flush = True)
    rec["cont"] = torch.tensor(cont, dtype = torch.int64)

save_file({k: v.contiguous() for k, v in rec.items()}, OUT)
print("saved", OUT)
