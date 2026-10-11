#!/usr/bin/env python3
"""Oracle fixture (test-time only): the image embeddings exllamav3's vision
tower produces for Qwen3.8-Flash-Next, for comparing Ampstein's tower output.

    python3 tests/oracle/make_qwen4_vision_oracle.py IMAGE [MODEL_DIR] [OUT]

Stores `emb` (n_tokens, hidden) fp16 and `grid` (the merged grid).
"""
import sys
import torch
from PIL import Image
from safetensors.torch import save_file
from exllamav3 import Config, Model, Tokenizer

img = sys.argv[1]
model_dir = sys.argv[2] if len(sys.argv) > 2 else "/data/models/Qwen3.8-Flash-Next-exl3-6.05bpw"
out = sys.argv[3] if len(sys.argv) > 3 else "qwen4_vision_oracle.safetensors"

config = Config.from_directory(model_dir)
vision = Model.from_config(config, component = "vision")
vision.load()
tok = Tokenizer.from_config(config)
emb = vision.get_image_embeddings(tokenizer = tok, image = Image.open(img))
e = emb.embeddings.detach().cpu().half().contiguous()
print("embeddings", tuple(e.shape))
save_file({"emb": e}, out)
