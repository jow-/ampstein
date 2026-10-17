#!/usr/bin/env python3
"""Attention oracle for DeepSeek-V4.1-Flash (tests/dsv41_attn.uc).

Runs DeepSeek's own reference attention (inference/model.py of deepseek-ai/DeepSeek-V4.1-Flash,
unmodified) on the CPU in fp32 for a few layers of the SAGE pack, with the TileLang kernels
replaced by plain-torch equivalents (the activation quantizers become identities; the FP8
weights are dequantized), and writes inputs and outputs to a safetensors fixture:

  x_L<i>   (T0 + ND, dim) fp32   the (already normed) attention input of layer i
  y_L<i>   (T0 + ND, dim) fp32   the attention output: rows [0, T0) from one prefill call, then ND decode steps

  python3 make_dsv41_oracle.py <ref_dir> <weights.safetensors> <out.safetensors> [T0 ND layers cand_blocks cand_block_size]

<ref_dir>: DeepSeek-V4.1-Flash/inference (model.py engram.py config.json), <weights>: a safetensors file
with the attn.* tensors of the layers, e.g. built by range-fetching them from the pack.
Needs torch (CPU is fine), safetensors, numpy, sympy.
"""
import json
import sys
import types

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ref_dir, wfile, out = sys.argv[1:4]
T0 = int(sys.argv[4]) if len(sys.argv) > 4 else 700
ND = int(sys.argv[5]) if len(sys.argv) > 5 else 6
LAYERS = [int(x) for x in (sys.argv[6] if len(sys.argv) > 6 else "2,3,20,21,24").split(",")]
CAND_BLOCKS = int(sys.argv[7]) if len(sys.argv) > 7 else 4
CAND_BS = int(sys.argv[8]) if len(sys.argv) > 8 else 8

sys.path.insert(0, ref_dir)

# ---- stand-ins for the TileLang kernels and the modules model.py imports but attention does not use
kernel = types.ModuleType("kernel")


def ident(x, *a, **k):
    return x


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, s, h, d = q.shape
    o = torch.empty_like(q)
    for s0 in range(0, s, 128):
        qs, ti = q[:, s0:s0 + 128], topk_idxs[:, s0:s0 + 128]
        idx = ti.long().clamp(min=0)
        g = kv[torch.arange(b)[:, None, None], idx]                      # (b, s', k, d)
        sc = torch.einsum("bshd,bskd->bshk", qs, g) * softmax_scale
        sc = sc.masked_fill((ti < 0)[:, :, None, :], float("-inf"))
        lg = torch.cat([sc, attn_sink.view(1, 1, h, 1).expand(b, sc.size(1), h, 1)], -1)
        p = torch.softmax(lg, -1)[..., :-1]
        o[:, s0:s0 + 128] = torch.einsum("bshk,bskd->bshd", p, g)
    return o


import os

QUANT = os.environ.get("DSV41_QUANT") == "1"      # emulate the reference's activation / KV quantizers


def fp8_fake(x, block=32):
    """act_quant(..., inplace=True) with power-of-two scales: fp8 e4m3 per block, dequantized."""
    sh = x.shape
    z = x.float().reshape(-1, block)
    amax = z.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    sc = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    return ((z / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc).reshape(sh)


def fp4_round(v):
    a = v.abs()
    r = torch.zeros_like(a)
    for thr, val, strict in ((0.25, 0.5, True), (0.75, 1.0, False), (1.25, 1.5, True), (1.75, 2.0, False),
                             (2.5, 3.0, True), (3.5, 4.0, False), (5.0, 6.0, True)):
        r = torch.where((a > thr) if strict else (a >= thr), torch.full_like(a, val), r)
    return torch.sign(v) * r


def fp4_fake(x, block=32, scale_dtype=torch.float8_e8m0fnu):
    sh = x.shape
    z = x.float().reshape(-1, block)
    amax = z.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp(min=6 * 2 ** -9)
        sc = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp(min=6 * 2 ** -126)
        sc = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    return (fp4_round((z / sc).clamp(-6, 6)) * sc).reshape(sh)


def q_act_quant(x, block, scale_fmt, scale_dtype, inplace=False):
    if QUANT:
        x.copy_(fp8_fake(x, block))
    return x


def q_fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    if QUANT:
        x.copy_(fp4_fake(x, block_size, scale_dtype))
    return x


kernel.act_quant = q_act_quant
kernel.fp4_act_quant = q_fp4_act_quant
kernel.fp4_gemm = None
kernel.fp8_gemm = None
kernel.hc_split_sinkhorn = None
kernel.sparse_attn = sparse_attn
sys.modules["kernel"] = kernel
ip = types.ModuleType("image_processor")
ip.IMAGE, ip.IMAGE_END, ip.IMAGE_NEW_LINE, ip.IMAGE_START = 0, 1, 2, 3
sys.modules["image_processor"] = ip
vis = types.ModuleType("vision")
vis.Aligner = vis.ViT = object
sys.modules["vision"] = vis

import model as M  # noqa: E402

torch.set_default_dtype(torch.float32)


def e8m0(t):
    return torch.pow(2.0, t.view(torch.uint8).float() - 127.0)


def linear(x, weight, bias=None):
    if QUANT and weight.dtype == torch.float8_e4m3fn:
        x = fp8_fake(x)
    if weight.dtype == torch.float8_e4m3fn:
        n, k = weight.shape
        s = e8m0(weight.scale)
        w = weight.float().view(n // 32, 32, k // 32, 32) * s.view(n // 32, 1, k // 32, 1)
        return F.linear(x.float(), w.view(n, k))
    return F.linear(x.float(), weight.float())


M.linear = linear
M.world_size, M.rank = 1, 0

cfgj = json.load(open(f"{ref_dir}/config.json"))
args = M.ModelArgs(
    max_batch_size=1, max_seq_len=T0 + ND + 8, dim=cfgj["dim"], moe_inter_dim=cfgj["moe_inter_dim"],
    n_layers=cfgj["n_layers"], n_heads=cfgj["n_heads"], q_lora_rank=cfgj["q_lora_rank"],
    head_dim=cfgj["head_dim"], rope_head_dim=cfgj["rope_head_dim"], o_groups=cfgj["o_groups"],
    o_lora_rank=cfgj["o_lora_rank"], window_size=cfgj["window_size"],
    compress_ratios=tuple(cfgj["compress_ratios"]), kv_source_layers=tuple(cfgj["kv_source_layers"]),
    index_source_layers=tuple(cfgj["index_source_layers"]),
    compress_rope_theta=cfgj["compress_rope_theta"], original_seq_len=cfgj["original_seq_len"],
    rope_theta=cfgj["rope_theta"], rope_factor=cfgj["rope_factor"], beta_fast=cfgj["beta_fast"],
    beta_slow=cfgj["beta_slow"], index_n_heads=cfgj["index_n_heads"], index_head_dim=cfgj["index_head_dim"],
    index_topk=cfgj["index_topk"], candidate_source_layer=cfgj["candidate_source_layer"],
    candidate_topk_blocks=CAND_BLOCKS, candidate_block_size=CAND_BS, norm_eps=cfgj["norm_eps"],
)

f = safe_open(wfile, "pt")


def load(mod, prefix):
    for name, p in list(mod.named_parameters()) + [(n, b) for n, b in mod.named_buffers() if False]:
        key = f"{prefix}.{name}"
        t = f.get_tensor(key)
        if p.dtype == torch.float8_e4m3fn:
            p.data = t
            sc = f.get_tensor(key[:-len("weight")] + "scale")
            p.scale = sc
            continue
        p.data = t.to(p.dtype if p.dtype != torch.bfloat16 else torch.float32)


torch.manual_seed(1234)
mods, X, Y = {}, {}, {}
for lid in LAYERS:
    attn = M.Attention(lid, args)
    # fp8 Linears keep their scale on the weight; bf16 ones are promoted to fp32
    for name, mod in attn.named_modules():
        if isinstance(mod, M.Linear):
            w = f.get_tensor(f"layers.{lid}.attn.{name}.weight")
            if name == "wo_a":      # the reference einsums bf16 weights here (convert.py dequantizes them)
                n, k = w.shape
                sc = e8m0(f.get_tensor(f"layers.{lid}.attn.{name}.scale"))
                dq = (w.float().view(n // 32, 32, k // 32, 32) * sc.view(n // 32, 1, k // 32, 1)).view(n, k)
                mod.weight = torch.nn.Parameter(dq, requires_grad=False)
            elif w.dtype == torch.float8_e4m3fn:
                mod.weight = torch.nn.Parameter(w, requires_grad=False)
                mod.weight.scale = f.get_tensor(f"layers.{lid}.attn.{name}.scale")
            else:
                mod.weight = torch.nn.Parameter(w.float(), requires_grad=False)
        elif isinstance(mod, M.RMSNorm):
            mod.weight = torch.nn.Parameter(f.get_tensor(f"layers.{lid}.attn.{name}.weight").float(), requires_grad=False)
    attn.attn_sink = torch.nn.Parameter(f.get_tensor(f"layers.{lid}.attn.attn_sink").float(), requires_grad=False)
    attn.eval()
    mods[lid] = attn
    X[lid] = torch.randn(1, T0 + ND, args.dim)
    Y[lid] = []

# the layers share state (pools, picks, candidates), so each call runs them in order
extra = {}
with torch.inference_mode():
    for lid in LAYERS:
        Y[lid].append(mods[lid](X[lid][:, :T0], 0)[0])
        if lid == args.candidate_source_layer:
            extra["cand_prefill"] = M.shared_attn.candidates[0].to(torch.uint8).contiguous()
    for i in range(ND):
        for lid in LAYERS:
            Y[lid].append(mods[lid](X[lid][:, T0 + i:T0 + i + 1], T0 + i)[0])

res = dict(extra)
for lid in LAYERS:
    res[f"x_L{lid}"] = X[lid][0].contiguous()
    res[f"y_L{lid}"] = torch.cat(Y[lid], 0).contiguous()
    print("layer", lid, "ratio", args.compress_ratios[lid], "y rms", res[f"y_L{lid}"].pow(2).mean().sqrt().item(), flush=True)

save_file(res, out)
print("wrote", out)
