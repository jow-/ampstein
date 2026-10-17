#!/usr/bin/env python3
"""Engram gate oracle for DeepSeek-V4.1-Flash (tests/dsv41_engram_apply.uc): DeepSeek's Engram.forward
(inference/model.py) on random streams and random table rows, with the real layer-1 projection.

  python3 make_dsv41_engram_apply_oracle.py <ref_dir> <weights.safetensors> <out.safetensors>

<weights>: a safetensors file with layers.1.engram.{wkv.weight,wkv.scale,q_weight,k_weight}.
"""
import sys
import types

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ref_dir, wfile, out = sys.argv[1:4]
sys.path.insert(0, ref_dir)
kernel = types.ModuleType("kernel")
kernel.act_quant = kernel.fp4_act_quant = kernel.fp4_gemm = kernel.fp8_gemm = kernel.sparse_attn = kernel.hc_split_sinkhorn = None
sys.modules["kernel"] = kernel
ip = types.ModuleType("image_processor")
ip.IMAGE, ip.IMAGE_END, ip.IMAGE_NEW_LINE, ip.IMAGE_START = 0, 1, 2, 3
sys.modules["image_processor"] = ip
vis = types.ModuleType("vision")
vis.Aligner = vis.ViT = object
sys.modules["vision"] = vis
import model as M  # noqa: E402
from engram import EngramLayout  # noqa: E402

torch.set_default_dtype(torch.float32)


def linear(x, weight, bias=None):
    n, k = weight.shape
    s = torch.pow(2.0, weight.scale.view(torch.uint8).float() - 127.0)
    w = weight.float().view(n // 32, 32, k // 32, 32) * s.view(n // 32, 1, k // 32, 1)
    return F.linear(x.float(), w.view(n, k))


M.linear = linear
M.world_size, M.rank = 1, 0
f = safe_open(wfile, "pt")
args = M.ModelArgs(dim=5120, hc_mult=4, norm_eps=1e-20)
layout = EngramLayout(max_ngram_size=4, layer_ids=(1,), num_embeddings=(1000,), primes=(((1,) * 8,) * 3,), n_heads=8, head_dim=256)
eng = M.Engram(args, 1, layout)
eng.wkv.weight = torch.nn.Parameter(f.get_tensor("layers.1.engram.wkv.weight"), requires_grad=False)
eng.wkv.weight.scale = f.get_tensor("layers.1.engram.wkv.scale")
eng.q_weight = torch.nn.Parameter(f.get_tensor("layers.1.engram.q_weight").float(), requires_grad=False)
eng.k_weight = torch.nn.Parameter(f.get_tensor("layers.1.engram.k_weight").float(), requires_grad=False)
torch.manual_seed(11)
T, D = 9, 5120
h = torch.randn(1, T, 4, D) * 2.0
emb = (torch.randn(1, T, 24, 256) * 0.4)


class FixedEmbed(torch.nn.Module):          # the table lookup is covered by the hash / gather tests
    def forward(self, ids):
        return emb


eng.embed = FixedEmbed()
with torch.inference_mode():
    y = eng(h, torch.zeros(1, T, 24, dtype=torch.int64))
save_file({"h": h[0].contiguous(), "emb": emb[0].reshape(T, 24 * 256).contiguous(), "y": y[0].contiguous()}, out)
print("wrote", out)
