#!/usr/bin/env python3
"""Oracle (test-time only): greedy continuation of an image+text chat
turn with exllamav3's own vision path, for comparing against Ampstein's
server output.

    .venv/bin/python tests/oracle/vision_generate_exllamav3.py IMAGE "question" [N] [MODEL_DIR]

Prints the greedy completion (thinking disabled, same template as the
server) and the prompt length."""
import sys, json
import torch
from PIL import Image
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

img_path, question = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
model_dir = sys.argv[4] if len(sys.argv) > 4 else "/data/models/Qwen3.8-27B-exl3-5.00bpw"

config = Config.from_directory(model_dir)
vision = Model.from_config(config, component = "vision")
vision.load()
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 8192)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model = model, cache = cache, tokenizer = tok)

emb = vision.get_image_embeddings(tokenizer = tok, image = Image.open(img_path))
prompt = ("<|im_start|>user\n" + emb.text_alias + question +
          "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
ids = tok.encode(prompt, encode_special_tokens = True, embeddings = [emb])
job = Job(input_ids = ids, max_new_tokens = n, decode_special_tokens = True,
          embeddings = [emb], sampler = GreedySampler())
gen.enqueue(job)
text = ""
while gen.num_remaining_jobs():
    for r in gen.iterate():
        text += r.get("text", "")
print(text)
print("PROMPT_TOKENS", ids.shape[-1])
