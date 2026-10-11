#!/usr/bin/env python3
"""Oracle fixture (test-time only, not shipping): Qwen3.8-Flash-Next
(Qwen4ExpForConditionalGeneration) reference activations captured from
exllamav3's own module chain.

Loads the whole text model (autosplit over all visible GPUs; stop other GPU
users first), runs ONE cache-less prefill and records, per transformer block i:

    b{i}_attn_in   f16 (T, hid)    sublayer input (after the hc collapse)
    b{i}_attn_out  f32 (T, hid)    sublayer output (before the residual update)
    b{i}_mlp_in / b{i}_mlp_out     same for the MoE
    b{i}_x         f32 (T, 4, hid) residual streams after the block
    b{i}_hc_attn_post / b{i}_hc_mlp_post   (T, 4) f32 inject gates
    b{i}_route_idx i64 (T, 10), b{i}_route_w f32 (T, 10)

plus  ids, h_embed, ple_emb (n-gram embedding), ple_in/ple_out (stream stacks
around the PLE layer), h_final (post-mixer f16), logits (f16, (T, V)) and a
greedy continuation `cont`.

    python3 tests/oracle/make_qwen4_oracle.py [MODEL_DIR] [OUT] [PROMPT_FILE]

With a PROMPT_FILE the prompt text is read from it (long-context runs that
exercise the QSA selection, > 2048 tokens) and the fixture also records the
indexer selection per full-attention layer (`b{i}_qsa_sel`, last 8 rows).
"""
import os, sys
import torch
from safetensors.torch import save_file
from exllamav3 import Config, Model, Tokenizer
from exllamav3.modules.transformer import TransformerBlock
from exllamav3.modules.ple import PLELayer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/models/Qwen3.8-Flash-Next-exl3-6.05bpw"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "qwen4_oracle.safetensors")
PROMPT_FILE = sys.argv[3] if len(sys.argv) > 3 else None
PROMPT = ("<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")
if PROMPT_FILE:
    PROMPT = open(PROMPT_FILE).read()
CONT = int(os.environ.get("CONT", "24"))

os.makedirs(os.path.dirname(OUT), exist_ok = True)

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
tok = Tokenizer.from_config(config)
rec = {}
armed = {"on": False}


def put(name, t):
    if armed["on"] and t is not None:
        rec[name] = t.detach().cpu().contiguous()


def wrap(obj, attr, fn):
    orig = getattr(obj, attr)

    def w(*a, **k):
        return fn(orig, *a, **k)
    setattr(obj, attr, w)


blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
ples = [m for m in model.modules if isinstance(m, PLELayer)]
print("blocks:", len(blocks), "ple layers:", len(ples))

for blk in blocks:
    i = blk.layer_idx

    def mk_forward(i):
        def f(orig, x, params, *a, **k):
            y = orig(x, params, *a, **k)
            put(f"b{i}_x", y)
            return y
        return f
    wrap(blk, "forward", mk_forward(i))

    def mk_sub(i, tag):
        def f(orig, x, params, *a, **k):
            put(f"b{i}_{tag}_in", x)
            y = orig(x, params, *a, **k)
            put(f"b{i}_{tag}_out", y)
            return y
        return f
    wrap(blk.attn, "forward", mk_sub(i, "attn"))
    wrap(blk.mlp, "forward", mk_sub(i, "mlp"))

    for tag, hc in (("attn", blk.attn_hc), ("mlp", blk.mlp_hc)):
        def mk_hc(i, tag):
            def f(orig, x, params, *a, **k):
                post, comb, coll = orig(x, params, *a, **k)
                put(f"b{i}_hc_{tag}_post", post)
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

    qsa = getattr(blk.attn, "qsa_indexer", None)
    if qsa is not None:
        def mk_sel(i, qsa):
            orig = qsa.select_indices

            def f(q_idx, pooled, past_len, batch_stride):
                out = orig(q_idx, pooled, past_len, batch_stride)
                put(f"b{i}_qsa_sel", out[-8:])
                return out
            return f
        qsa.select_indices = mk_sel(i, qsa)

for p in ples:
    def mk_ple(orig, x, params, *a, **k):
        put("ple_in", x)
        y = orig(x, params, *a, **k)
        put("ple_out", y)
        return y
    wrap(p, "forward", mk_ple)

    def mk_emb(orig, x, params, *a, **k):
        y = orig(x, params, *a, **k)
        put("ple_emb", y)
        return y
    wrap(p.ple_embedding, "forward", mk_emb)

if os.environ.get("DRY"):
    ids = tok.encode(PROMPT, encode_special_tokens = True)
    print("DRY ok, ids", ids.shape, ids.flatten().tolist()[:12])
    sys.exit(0)

print("loading (autosplit)...", flush = True)
model.load(progressbar = True)

ids = tok.encode(PROMPT, encode_special_tokens = True)
print("ids", tuple(ids.shape), ids.flatten().tolist()[:40], flush = True)

with torch.inference_mode():
    armed["on"] = True
    logits = model.forward(ids, params = {"attn_mode": "flash_attn_nc"})
    armed["on"] = False
    print("logits", tuple(logits.shape), flush = True)

rec["ids"] = ids.reshape(-1).to(torch.int64).cpu()
rec["logits"] = logits[0].half().cpu()

emb = model.modules[0]
with torch.inference_mode():
    rec["h_embed"] = emb.forward(ids.to(emb.device), {}).half().cpu()

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
