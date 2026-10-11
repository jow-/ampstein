#!/usr/bin/env python3
"""Oracle (test-time only, not shipping): depth-resolved quality fixture for
Qwen3.8-Flash-Next on ONE long real text (QSA sparse path active everywhere).

The model prefills the text in CHUNK-token chunks through a cache (recurrent
states carried) and, at each checkpoint depth D, the logits of the last ROWS positions are recorded:

    ids         i64 (Tmax,)
    D{D}_logits f16 (ROWS, V)    logits at positions D-ROWS .. D-1
    D{D}_nll    f32 (ROWS-1,)    -log p(ids[t+1]) for t in D-ROWS .. D-2

    python3 tests/oracle/make_qwen4_depth_oracle.py MODEL OUT TEXT_FILE [DEPTHS]

Stop other GPU users first. DEPTHS is a comma list (default 8192,32768,65536,98304,131072).
"""
import os, sys
import torch
from safetensors.torch import save_file
from exllamav3 import Config, Model, Tokenizer, Cache

MODEL_DIR = sys.argv[1]
OUT = sys.argv[2]
TEXT = sys.argv[3]
DEPTHS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "8192,32768,65536,98304,131072").split(",")]
ROWS = 256
CHUNK = int(os.environ.get("CHUNK", 2048))

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
tok = Tokenizer.from_config(config)
ids = tok.encode(open(TEXT, encoding = "utf8").read(), add_bos = False)
T = ids.shape[-1]
print("tokens:", T, flush = True)
assert T >= max(DEPTHS), "text too short for the deepest checkpoint"
ids = ids[:, :max(DEPTHS)]
if os.environ.get("DRY"):
    sys.exit(0)

assert all(D % CHUNK == 0 for D in DEPTHS)
dmax = max(DEPTHS)
cache = Cache(model, max_num_tokens = dmax + 256)
model.load(progressbar = True)
rec = {"ids": ids[0].to(torch.int64).cpu().contiguous()}
prev = {}
cap = -(-(dmax + 64) // 256) * 256
params = {}
with torch.inference_mode():
    for c0 in range(0, dmax, CHUNK):
        c1 = c0 + CHUNK
        want = c1 in DEPTHS
        params = {
            "attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, cap), "past_len": c0,
            "last_tokens_only": ROWS if want else 1,
        }
        if "recurrent_states" in prev:
            params["recurrent_states"] = prev["recurrent_states"]
        lg = model.forward(ids[:, c0:c1], params = params)[0]
        prev = params
        if want:
            D = c1
            lp = torch.log_softmax(lg.float(), dim = -1)
            tgt = ids[0, D - ROWS + 1:D].to(lp.device)
            nll = -lp[:-1].gather(1, tgt.view(-1, 1)).view(-1)
            print("depth %6d: mean NLL %.5f" % (D, nll.mean().item()), flush = True)
            rec[f"D{D}_logits"] = lg.half().cpu().contiguous()
            rec[f"D{D}_nll"] = nll.float().cpu().contiguous()
            del lp
        del lg
        if (c0 // CHUNK) % 8 == 7: print("  %d tokens" % c1, flush = True)

save_file(rec, OUT)
print("saved", OUT)
