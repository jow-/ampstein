#!/usr/bin/env python3
"""Oracle fixture (test-time only, not shipping): DeepSeek-V4-Flash-Vision-Exp tower
reference from exllamav3 (run with a python environment that has exllamav3 installed).

    dsv4_vision_oracle_<name>.safetensors:
        patches  f16 (N, 588)      preprocessed raster patches
        dims     i64 (8,)          nvh, nvw, nh, nw, orig_w, orig_h, lead, len(block)
        b0, bl   f16 (N, 1024)     hidden after block 0 / the last block
        unf      f16 (n, 9216)     aligner input (after norm + unfold)
        out      f16 (nh*nw, 4096) aligner output
        block    f16 (len, 4096)   whole N-layout embedding block (start_pos 0 -> 3 lead pads)
        types    i64 (len,)        row types of the block

    python make_dsv4_vision_oracle.py MODEL_DIR OUTDIR
"""
import os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file
from exllamav3 import Model, Config

MODEL_DIR = sys.argv[1]
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok = True)

def make(name, W, H, mode):
    yy, xx = np.mgrid[0:H, 0:W]
    arr = np.stack([(xx * 255 // W), (yy * 255 // H), ((xx + yy) * 255 // (W + H))], -1).astype(np.uint8)
    img = Image.fromarray(arr, "RGB").convert(mode)
    d = ImageDraw.Draw(img)
    d.ellipse((W // 8, H // 8, W // 2, H // 2), fill = (220, 30, 30, 255), outline = (255, 255, 255, 255), width = 3)
    d.rectangle((W // 2, H // 2, W * 9 // 10, H * 9 // 10), fill = (20, 30, 200, 255))
    d.text((W // 3, H // 10), "ucode", fill = (255, 255, 0, 255))
    path = os.path.join(outdir, f"dsv4_vision_{name}.png")
    img.save(path)
    return path

images = {
    "odd": make("odd", 301, 223, "RGBA"),       # padded (not stretched), alpha
    "wide": make("wide", 1600, 150, "RGB"),     # aspect >= 8: stretched
}

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config, component = "vision")
model.load("cuda:0")

with torch.inference_mode():
    for name, path in images.items():
        img = Image.open(path)
        patches, nvh, nvw, nh, nw = model.preprocess(img)
        params = {"causal": False, "grid_hw": (nvh, nvw)}
        from exllamav3.architecture.mm_processing.qwen2 import qwen2_position_embedding_grid_2d
        v = config.vision
        params["inv_freq"] = qwen2_position_embedding_grid_2d((1, nvh, nvw), v.head_dim, 1, v.rope_theta)
        x = patches.unsqueeze(0).to("cuda:0")
        x = model.modules[0].forward(x, params)
        mods = model.modules[1:1 + v.n_layers]
        x = mods[0].forward(x, params)
        b0 = x[0].half().cpu()
        for m in mods[1:]:
            x = m.forward(x, params)
        bl = x[0].half().cpu()
        xn = model.modules[1 + v.n_layers].forward(x, params)
        al = model.aligner
        unf = al.unshuffle(xn.view(nvh * nvw, -1).half(), nvh, nvw).cpu()
        out = model.encode_patches(patches, nvh, nvw).cpu()
        mme = model.get_image_embeddings(None, img)
        block = mme.embeddings.half().cpu()
        types = torch.tensor(mme.metadata["block_types"], dtype = torch.int64)
        lead = mme.align_lead
        print(name, "patches", tuple(patches.shape), "grid vit", (nvh, nvw), "llm", (nh, nw), "block", tuple(block.shape), "lead", lead)
        save_file({
            "patches": patches.half().contiguous().cpu(),
            "dims": torch.tensor([nvh, nvw, nh, nw, img.width, img.height, lead, block.shape[0]], dtype = torch.int64),
            "b0": b0.contiguous(), "bl": bl.contiguous(), "unf": unf.contiguous(),
            "out": out.reshape(nh * nw, -1).contiguous(), "block": block.contiguous(), "types": types,
        }, os.path.join(outdir, f"dsv4_vision_oracle_{name}.safetensors"))
