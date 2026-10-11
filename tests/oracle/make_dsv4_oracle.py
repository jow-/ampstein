#!/usr/bin/env python3
"""Oracle fixtures (test-time only, not shipping): DeepSeek-V4-Flash reference
activations captured from exllamav3's own module chain.

Loads the whole text model (autosplit over all visible GPUs; ~115 GB, stop
other GPU users first) and runs cache-less prefills (attn_mode flash_attn_nc,
the stateless HF-cache=None semantics) of three prompts:

  short   a chat prompt (~25 tokens): every tensor, full logits, greedy
          continuation. HCA/CSA pools are (almost) empty here.
  medium  ~700 tokens of text: HCA pools (128-token windows) and CSA pools
          are populated; dense pool attention (<= index_topk entries).
  long    ~2600 tokens: CSA pools exceed index_topk (512) -> the lightning
          indexer top-k selection is active.

Per transformer block i (all prompts; long/medium keep only the LAST `KEEP`
rows of every per-token tensor to bound the file size):

    b{i}_attn_in   f16 (T, hid)    attention input (after hc collapse + norm)
    b{i}_attn_out  f32 (T, hid)    attention output (before the residual update)
    b{i}_mlp_in    f16 (T, hid)    ffn input
    b{i}_mlp_out   f32 (T, hid)    ffn output
    b{i}_x         f32 (T, 4, hid) residual streams after the block
    b{i}_hc_attn_post/comb, b{i}_hc_ffn_post/comb
    b{i}_route_idx i64 (T, 6), b{i}_route_w f32 (T, 6)

Attention internals for INTERNAL_LAYERS (dsa_attn inputs/outputs):

    a{i}_q         f16 (T, 64, 512)  query after q_b / head norm / rope
    a{i}_kv        f16 (T, 512)      kv rows after kv_norm / rope
    a{i}_pool_c    f16 (E, 448)      compressed pool, nope part
    a{i}_pool_r    f16 (E, 64)       compressed pool, rope part
    a{i}_out       f16 (8, T, 4096)  attention output, group-major, de-rotated
    a{i}_idx       i32 (T, K)        indexer selection (CSA, top-k regime)

plus  ids, h_embed, logits (f16), and for `short` a greedy continuation `cont`.
Each prompt goes to its own file  <OUT_PREFIX>_{short,medium,long}.safetensors.

    python3 tests/oracle/make_dsv4_oracle.py [MODEL_DIR] [OUT_PREFIX] [TEXT_FILE]
"""
import os, sys
import torch
from safetensors.torch import save_file
from exllamav3 import Config, Model, Tokenizer
from exllamav3.modules.transformer import TransformerBlock
import exllamav3.modules.dsv4 as dsv4mod

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw"
OUT_PREFIX = sys.argv[2] if len(sys.argv) > 2 else "dsv4_oracle"      # output file prefix
if len(sys.argv) < 4:
    sys.exit("usage: make_dsv4_oracle.py MODEL_DIR OUT_PREFIX TEXT_FILE  (TEXT_FILE: a long plain-text file)")
TEXT_FILE = sys.argv[3]
CONT = 24
KEEP = 128
HEAD = 8
ONLY = os.environ.get("ONLY")      # e.g. ONLY=medium: run just that prompt
INTERNAL_LAYERS = {0, 1, 2, 3, 4}
BOS = "<｜begin▁of▁sentence｜>"
CHAT = BOS + "<｜User｜>What is the capital of France? Answer in one sentence.<｜Assistant｜></think>"

os.makedirs(os.path.dirname(OUT_PREFIX), exist_ok = True)

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
tok = Tokenizer.from_config(config)
rec = {}
state = {"on": False, "T": 0, "keep": None, "layer": -1}


def put(name, t, rows = True):
    if not state["on"]:
        return
    t = t.detach()
    if rows and state["keep"] and t.dim() >= 2 and t.shape[1] == state["T"]:
        # long prompts: the last KEEP rows, plus the first HEAD rows under name@h (the
        # attention-sink / massive-activation tokens live there)
        rec[name + "@h"] = t[:, :HEAD].cpu().contiguous()
        t = t[:, -state["keep"]:]
    rec[name] = t.cpu().contiguous()


def wrap(obj, attr, fn):
    orig = getattr(obj, attr)
    setattr(obj, attr, lambda *a, **k: fn(orig, *a, **k))


# dsa_attn is looked up as a module global at call time
_orig_dsa_attn = dsv4mod.dsa_attn


def dsa_attn_rec(q, pool_c, pool_r, bt, *a, **k):
    out = _orig_dsa_attn(q, pool_c, pool_r, bt, *a, **k)
    i = state["layer"]
    if state["on"] and i in INTERNAL_LAYERS:
        T = q.shape[0]
        keep = state["keep"] or T
        put(f"a{i}_q", q[-keep:], rows = False)
        kvc = k.get("kv_chunk")
        if kvc is not None:
            put(f"a{i}_kv", kvc[-keep:], rows = False)
        npool = k.get("pool_len", 0)
        if npool:
            put(f"a{i}_pool_c", pool_c.reshape(-1, pool_c.shape[-1])[:npool], rows = False)
            put(f"a{i}_pool_r", pool_r.reshape(-1, pool_r.shape[-1])[:npool], rows = False)
        put(f"a{i}_out", out[:, -keep:], rows = False)
        if k.get("indices") is not None:
            put(f"a{i}_idx", k["indices"][-keep:], rows = False)
    return out


dsv4mod.dsa_attn = dsa_attn_rec

blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
print("blocks:", len(blocks))

for blk in blocks:
    i = blk.layer_idx

    def mk_forward(i):
        def f(orig, x, params, *a, **k):
            state["layer"] = i
            y = orig(x, params, *a, **k)
            put(f"b{i}_x", y)
            return y
        return f
    wrap(blk, "forward", mk_forward(i))

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
                if state["on"]:
                    kk = state["keep"] or sel.shape[0]
                    rec[f"b{i}_route_idx"] = sel[-kk:].detach().cpu().to(torch.int64)
                    rec[f"b{i}_route_w"] = w[-kk:].detach().float().cpu()
                return sel, w
            return f
        blk.mlp.routing_fn = mk_route(i, rf)

# prompts
text = open(TEXT_FILE, encoding = "utf-8").read() if os.path.exists(TEXT_FILE) else ("lorem ipsum " * 2000)
tok_text = tok.encode(text, encode_special_tokens = False).flatten()
bos = tok.encode(BOS, encode_special_tokens = True).flatten()[:1]
prompts = {
    "short": tok.encode(CHAT, encode_special_tokens = True),
    "medium": torch.cat([bos, tok_text[:700]]).unsqueeze(0),
    "long": torch.cat([bos, tok_text[:2600]]).unsqueeze(0),
}

if os.environ.get("DRY"):
    for n, p in prompts.items():
        print("DRY", n, tuple(p.shape), p.flatten().tolist()[:10])
    sys.exit(0)

print("loading (autosplit)...", flush = True)
model.load(progressbar = True)

for name, ids in prompts.items():
    if ONLY and name != ONLY:
        continue
    rec.clear()
    T = ids.shape[1]
    state.update(T = T, keep = None if name == "short" else KEEP, layer = -1)
    print(name, "ids", tuple(ids.shape), flush = True)
    with torch.inference_mode():
        state["on"] = True
        logits = model.forward(ids, params = {"attn_mode": "flash_attn_nc"})
        state["on"] = False
    print("logits", tuple(logits.shape), flush = True)
    rec["ids"] = ids.reshape(-1).to(torch.int64).cpu()
    rec["logits"] = (logits[0] if name == "short" else logits[0, -64:]).half().cpu()
    emb = model.modules[0]
    with torch.inference_mode():
        h = emb.forward(ids.to(emb.device), {}).half().cpu()
    rec["h_embed"] = h if name == "short" else h[:, -KEEP:]

    if name == "short":
        cont, work = [], ids.clone()
        with torch.inference_mode():
            for _ in range(CONT):
                lg = model.forward(work, params = {"attn_mode": "flash_attn_nc"})
                nxt = int(lg[0, -1].argmax())
                cont.append(nxt)
                work = torch.cat([work, torch.tensor([[nxt]], dtype = work.dtype)], dim = 1)
        print("cont", cont, flush = True)
        rec["cont"] = torch.tensor(cont, dtype = torch.int64)

    out = f"{OUT_PREFIX}_{name}.safetensors"
    save_file(rec, out)
    print("saved", out, len(rec), "tensors", flush = True)
