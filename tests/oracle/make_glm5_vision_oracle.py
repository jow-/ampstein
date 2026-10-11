#!/usr/bin/env python3
"""Oracle fixture (test-time only, not shipping): GLM-5.3-Flash vision tower
reference from exllamav3.

    tests/fixtures/vision_test.png        deterministic test image (RGBA,
                                          odd size: exercises alpha
                                          compositing + smart resize)
    tests/fixtures/glm5_vision_oracle.safetensors :
        patches  f16 (N, 1536)    preprocessed, flattened patches
        grid     i64 (3,)         (t, h, w) patch grid
        block0   f16 (N, 1024)    hidden after block 0 (fp32 stream -> f16)
        emb      f16 (N/4, 4096)  final merger output

Run from repo root (CUDA, ~1 GiB free):
    .venv/bin/python tests/oracle/make_vision_oracle.py [MODEL_DIR]
"""
import os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file
from exllamav3 import Model, Config

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/models/GLM-5.3-Flash-exl3-3.05bpw"
outdir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(outdir, exist_ok = True)

# deterministic image: gradients + shapes + a translucent corner
W, H = 301, 223
yy, xx = np.mgrid[0:H, 0:W]
arr = np.stack([(xx * 255 // W), (yy * 255 // H), ((xx + yy) * 255 // (W + H))], -1).astype(np.uint8)
img = Image.fromarray(arr, "RGB").convert("RGBA")
d = ImageDraw.Draw(img)
d.ellipse((40, 30, 140, 130), fill = (220, 30, 30, 255), outline = (255, 255, 255, 255), width = 3)
d.rectangle((170, 90, 270, 190), fill = (20, 30, 200, 255))
d.polygon([(10, 200), (80, 150), (120, 210)], fill = (30, 200, 60, 255))
d.rectangle((0, 0, 60, 40), fill = (0, 0, 0, 90))
d.text((150, 20), "ucode", fill = (255, 255, 0, 255))
png = os.path.join(outdir, "vision_test.png")
if not os.path.exists(png):
    img.save(png)
img = Image.open(png)

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config, component = "vision")
model.load("cuda:0")

with torch.inference_mode():
    patches, new_size, grid = model.preprocess(img)
    print("preprocessed", tuple(patches.shape), new_size, grid)
    mme = model.get_image_embeddings(None, img)
    emb = mme.embeddings
    print("emb", tuple(emb.shape), emb.dtype)

    # hidden after block 0: run the module chain by hand
    from exllamav3.architecture.mm_processing.qwen2 import qwen2_position_embedding_grid_2d
    v = config.vision
    inv = qwen2_position_embedding_grid_2d(grid, v.head_dim, v.spatial_merge_size, v.rope_theta)
    params = { "causal": False, "grid_thw": torch.tensor([grid], dtype = torch.int), "inv_freq": inv }
    x = patches.unsqueeze(0).to("cuda:0")
    x = model.prepare_inputs(x, params) if hasattr(model, "prepare_inputs") else x
    for m in model.modules[:2]:
        x = m.forward(x, params)
    block0 = x[0].half().cpu()
    per_block = {"b0": block0.clone()}
    for bi, m in enumerate(model.modules[2:-3], start = 1):       # blocks 1..depth-1
        x = m.forward(x, params)
        per_block[f"b{bi}"] = x[0].half().cpu()
    blocks_out = x[0].half().cpu()
    for m in model.modules[-3:-1]:      # post_layernorm, downsample
        x = m.forward(x, params)
    down = x.reshape(-1, x.shape[-1]).half().cpu()
    print("module chain tail:", [type(m).__name__ for m in model.modules[-4:]])

save_file({
    "patches": patches.half().contiguous().cpu(),
    "grid": torch.tensor(grid, dtype = torch.int64),
    "block0": block0.contiguous(),
    **{k: v.contiguous() for k, v in per_block.items()},
    "blocks_out": blocks_out.contiguous(),
    "down": down.contiguous(),
    "emb": emb.half().contiguous().cpu(),
}, os.path.join(outdir, "glm5_vision_oracle.safetensors"))
print("saved.")
model.unload()
