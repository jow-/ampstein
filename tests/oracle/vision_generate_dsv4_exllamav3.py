#!/usr/bin/env python3
"""Oracle (test-time only): greedy continuation of an image+text DeepSeek-V4 turn with
exllamav3's own vision path, for comparing against Ampstein's server.

    python vision_generate_dsv4_exllamav3.py IMAGE PROMPT_FILE [N] [MODEL_DIR]

PROMPT_FILE holds the rendered prompt (bos + user turn + generation prompt) with the image
as the placeholder <｜rl_image_pad｜>, which is swapped for the embedding's text alias."""
import sys
import torch
from PIL import Image
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

img_path, prompt_file = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
model_dir = sys.argv[4] if len(sys.argv) > 4 else "/data/models/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw"

config = Config.from_directory(model_dir)
vision = Model.from_config(config, component = "vision")
vision.load()
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 8192)
model.load(progressbar = False)
tok = Tokenizer.from_config(config)
gen = Generator(model = model, cache = cache, tokenizer = tok)

emb = vision.get_image_embeddings(tokenizer = tok, image = Image.open(img_path))
prompt = open(prompt_file, encoding = "utf8").read()
ph = "<｜rl_image_pad｜>"
assert prompt.count(ph) == 1, "expected exactly one image placeholder"
prompt = prompt.replace(ph, emb.text_alias)
ids = tok.encode(prompt, encode_special_tokens = True, embeddings = [emb])
job = Job(input_ids = ids, max_new_tokens = n, decode_special_tokens = True,
          embeddings = [emb], sampler = GreedySampler(), return_top_tokens = 0)
gen.enqueue(job)
text = ""
while gen.num_remaining_jobs():
    for r in gen.iterate():
        text += r.get("text", "")
print(text)
print("PROMPT_TOKENS", ids.shape[-1])
