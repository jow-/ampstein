#!/usr/bin/env python3
"""Hyper-connection oracle for DeepSeek-V4.1-Flash (tests/dsv41_hc.uc): DeepSeek's Block.hc_mixes / hc_pre /
hc_post (inference/model.py) with the sinkhorn kernel in plain torch.

  python3 make_dsv41_hc_oracle.py <ref_dir> <weights.safetensors> <out.safetensors>

<weights>: a safetensors file with layers.2.hc_{attn,ffn}_{fn,base,scale}.
"""
import sys
import types

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ref_dir, wfile, out = sys.argv[1:4]
sys.path.insert(0, ref_dir)
kernel = types.ModuleType("kernel")
kernel.act_quant = kernel.fp4_act_quant = kernel.fp4_gemm = kernel.fp8_gemm = kernel.sparse_attn = None


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, -1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


kernel.hc_split_sinkhorn = hc_split_sinkhorn
sys.modules["kernel"] = kernel
ip = types.ModuleType("image_processor")
ip.IMAGE, ip.IMAGE_END, ip.IMAGE_NEW_LINE, ip.IMAGE_START = 0, 1, 2, 3
sys.modules["image_processor"] = ip
vis = types.ModuleType("vision")
vis.Aligner = vis.ViT = object
sys.modules["vision"] = vis
import model as M  # noqa: E402

f = safe_open(wfile, "pt")
blk = types.SimpleNamespace(norm_eps=1e-20, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
torch.manual_seed(5)
T, D = 12, 5120
x = torch.randn(1, T, 4, D) * 3.0
res = {"x": x[0].contiguous()}
for tag in ("attn", "ffn"):
    fn = f.get_tensor(f"layers.2.hc_{tag}_fn").float()
    base = f.get_tensor(f"layers.2.hc_{tag}_base").float()
    scale = f.get_tensor(f"layers.2.hc_{tag}_scale").float()
    pre, post, comb = M.Block.hc_mixes(blk, x, fn, scale, base)
    res[f"pre_{tag}"], res[f"post_{tag}"], res[f"comb_{tag}"] = pre[0].contiguous(), post[0].contiguous(), comb[0].contiguous()
    res[f"col_{tag}"] = M.Block.hc_pre(blk, x, pre)[0].contiguous()
save_file(res, out)
print("wrote", out)
