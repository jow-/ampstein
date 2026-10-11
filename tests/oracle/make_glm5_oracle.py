#!/usr/bin/env python3
"""Oracle fixture (test-time only, not shipping): GLM-5.3-Flash reference
activations captured from exllamav3's own module chain.

Loads the whole text model (autosplit over all visible GPUs; ~117 GB, stop
other GPU users first), runs ONE cache-less prefill of a short chat prompt
and records, per transformer block i:

    b{i}_attn_in   f16 (T, hid)    attention input (after hc collapse + norm)
    b{i}_attn_out  f32 (T, hid)    attention output (before the residual update)
    b{i}_mlp_in    f16 (T, hid)    ffn input (after hc collapse + norm)
    b{i}_mlp_out   f32 (T, hid)    ffn output
    b{i}_x         f32 (T, 4, hid) residual streams after the block
    b{i}_hc_attn_post/comb, b{i}_hc_ffn_post/comb   (T,4)/(T,4,4) f32
    b{i}_route_idx i64 (T, 8), b{i}_route_w f32 (T, 8)   (MoE blocks)

plus  ids, h_embed (f16), h_final (post-norm f16), logits (f16, (T, V)),
and a greedy continuation `cont` (N tokens, cache-less forwards).

    python3 tests/oracle/make_glm5_oracle.py [MODEL_DIR] [OUT]
"""
import os, sys
import torch
from safetensors.torch import save_file
from exllamav3 import Config, Model, Tokenizer
from exllamav3.modules.transformer import TransformerBlock

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/models/GLM-5.3-Flash-exl3-3.05bpw"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "glm5_oracle.safetensors")
PROMPT = "[gMASK]<sop><|user|>\nWhat is the capital of France? Answer in one sentence.<|assistant|>\n</think>"
CONT = 24

os.makedirs(os.path.dirname(OUT), exist_ok = True)

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
tok = Tokenizer.from_config(config)
rec = {}
armed = {"on": False}


def put(name, t):
    if armed["on"]:
        rec[name] = t.detach().cpu().contiguous()


def wrap(obj, attr, fn):
    orig = getattr(obj, attr)

    def w(*a, **k):
        r = fn(orig, *a, **k)
        return r
    setattr(obj, attr, w)


blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
print("blocks:", len(blocks))

for blk in blocks:
    i = blk.layer_idx

    def mk_forward(i, blk):
        def f(orig, x, params, *a, **k):
            y = orig(x, params, *a, **k)
            put(f"b{i}_x", y)
            return y
        return f
    wrap(blk, "forward", mk_forward(i, blk))

    def mk_attn(i):
        def f(orig, x, params, *a, **k):
            put(f"b{i}_attn_in", x)
            y = orig(x, params, *a, **k)
            put(f"b{i}_attn_out", y)
            return y
        return f
    wrap(blk.attn, "forward", mk_attn(i))

    def mk_mlp(i):
        def f(orig, x, params, *a, **k):
            put(f"b{i}_mlp_in", x)
            y = orig(x, params, *a, **k)
            put(f"b{i}_mlp_out", y)
            return y
        return f
    wrap(blk.mlp, "forward", mk_mlp(i))

    for tag, hc in (("attn", blk.attn_hc), ("ffn", blk.mlp_hc)):
        def mk_hc(i, tag):
            def f(orig, x, params, *a, **k):
                post, comb, coll = orig(x, params, *a, **k)
                put(f"b{i}_hc_{tag}_post", post)
                put(f"b{i}_hc_{tag}_comb", comb)
                return post, comb, coll
            return f
        wrap(hc, "mix", mk_hc(i, tag))

    rf = getattr(blk.mlp, "routing_fn", None)
    if rf is not None:
        def mk_route(i, rf):
            def f(bsz, cfg, y, params):
                sel, w = rf(bsz, cfg, y, params)
                put(f"b{i}_route_idx", sel)
                put(f"b{i}_route_w", w.float())
                return sel, w
            return f
        blk.mlp.routing_fn = mk_route(i, rf)

if os.environ.get("DRY"):
    ids = tok.encode(PROMPT, encode_special_tokens = True)
    print("DRY ok, ids", ids.shape, ids.flatten().tolist()[:12])
    sys.exit(0)

print("loading (autosplit)...", flush = True)
model.load(progressbar = True)

ids = tok.encode(PROMPT, encode_special_tokens = True)
print("ids", tuple(ids.shape), ids.flatten().tolist(), flush = True)

with torch.inference_mode():
    armed["on"] = True
    logits = model.forward(ids, params = {"attn_mode": "flash_attn_nc"})
    armed["on"] = False
    print("logits", tuple(logits.shape), flush = True)

rec["ids"] = ids.reshape(-1).to(torch.int64).cpu()
rec["logits"] = logits[0].half().cpu()

# final-norm output and embed output via the module chain (cheap re-run of
# the first / last modules)
emb = model.modules[0]
with torch.inference_mode():
    rec["h_embed"] = emb.forward(ids.to(emb.device), {}).half().cpu()

# greedy continuation: repeated cache-less forwards over the growing prefix
cont, work = [], ids.clone()
with torch.inference_mode():
    for _ in range(CONT):
        lg = model.forward(work, params = {"attn_mode": "flash_attn_nc"})
        nxt = int(lg[0, -1].argmax())
        cont.append(nxt)
        work = torch.cat([work, torch.tensor([[nxt]], dtype = work.dtype)], dim = 1)
print("cont", cont, flush = True)
rec["cont"] = torch.tensor(cont, dtype = torch.int64)

save_file(rec, OUT)
print("saved", OUT, len(rec), "tensors")
