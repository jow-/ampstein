#!/usr/bin/env python3
"""Oracle (test-time only): perplexity-style fixture for GLM-5.3-Flash on
real text longer than index_topk (sparse indexer active).

Text: the first ~3.6k tokens of README.md (any prose works). The model runs
a cache-less forward over all tokens; logits of the LAST 512 positions are
recorded together with the targets (the next tokens):

    ids      i64 (T,)
    logits   f16 (512, V)    logits at positions T-512 .. T-1
    nll      f32 (511,)      -log p(ids[t+1]) for t in T-512 .. T-2

    python3 tests/oracle/make_glm5_ppl_oracle.py [MODEL_DIR] [TEXT_FILE]
"""
import os, sys
import torch
from safetensors.torch import save_file
from exllamav3 import Config, Model, Tokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/models/GLM-5.3-Flash-exl3-3.05bpw"
HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "README.md")
OUT = os.path.join(HERE, "..", "fixtures", "glm5_ppl_oracle.safetensors")
N, ROWS = 3600, 512

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
tok = Tokenizer.from_config(config)
ids = tok.encode("[gMASK]<sop>" + open(TEXT, encoding = "utf8").read(), encode_special_tokens = True)[:, :N]
T = ids.shape[-1]
print("tokens:", T, flush = True)
assert T > 2048 + ROWS
if os.environ.get("DRY"):
    sys.exit(0)

model.load(progressbar = True)
with torch.inference_mode():
    lg = model.forward(ids, params = {"attn_mode": "flash_attn_nc", "last_tokens_only": ROWS})[0]
    lp = torch.log_softmax(lg.float(), dim = -1)
    tgt = ids[0, T - ROWS + 1:].to(lp.device)
    nll = -lp[:-1].gather(1, tgt.view(-1, 1)).view(-1)
    print("mean NLL %.5f  ppl %.4f" % (nll.mean().item(), nll.mean().exp().item()), flush = True)

save_file({"ids": ids[0].to(torch.int64).cpu(), "logits": lg.half().cpu().contiguous(),
           "nll": nll.float().cpu().contiguous()}, OUT)
print("saved", OUT)
