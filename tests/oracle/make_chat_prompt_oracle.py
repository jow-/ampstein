#!/usr/bin/env python3
"""Oracle fixture: a full llama-3 CHAT prompt, prefill logits.

The 8-token plain-prompt fixtures never exercise token 128000 (BOS):
its embedding row is tiny (~3e-3), which is what exposed the all-fp16
RMSNorm variance bug (mean(x^2) subnormal + eps rounding). This fixture
locks the chat-template path in:

    fixtures/chat_prompt_logits.safetensors :
        ids       int64   (S,)   BOS + atomized llama-3 chat template
        logits    float32 (S, V) exllamav3 Model.forward logits

The prompt is the rendered llama-3.2 template (default system preamble
+ one user turn + generation prompt), encoded with
encode_special_tokens=True (header markers atomized to 128006/128007).

Run from repo root (needs MODEL_DIR below + CUDA):
    .venv/bin/python tests/oracle/make_chat_prompt_oracle.py
"""
import os

import torch
from safetensors.torch import save_file
from exllamav3 import Config, Model, Tokenizer

MODEL_DIR = "/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw"

PROMPT = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "What is the capital of France?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

device = "cuda:0"
torch.cuda.set_device(0)

outdir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(outdir, exist_ok=True)

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
model.load()
tok = Tokenizer(config)

# encode_special_tokens=True: atomize <|start_header_id|> etc.; the
# leading <|begin_of_text|> becomes id 128000 through the same path
ids = tok.encode(PROMPT, add_bos=False, encode_special_tokens=True)
ids = ids.reshape(-1).tolist()
assert ids[0] == 128000 and ids[1] == 128006, "unexpected encoding"
print("token ids:", ids, "len:", len(ids))

input_ids = torch.tensor([ids], dtype=torch.long, device=device)
with torch.inference_mode():
    logits = model.forward(input_ids)

logits = logits[0].float().cpu()
print("logits:", tuple(logits.shape))

argmax = [int(logits[i].argmax()) for i in range(logits.shape[0])]
print("per-position argmax:", argmax)
print("greedy continuation would start with:",
      repr(tok.decode(torch.tensor([argmax[-1:]]))))

save_file({
    "ids": torch.tensor(ids, dtype=torch.int64),
    "logits": logits.to(torch.float16),
}, os.path.join(outdir, "chat_prompt_logits.safetensors"))
print("wrote", os.path.join(outdir, "chat_prompt_logits.safetensors"))