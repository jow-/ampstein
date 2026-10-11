#!/usr/bin/env python3
"""Oracle fixture: per-layer hidden states + final logits from a REAL
model dir, captured through exllamav3's own module chain.

Runs turboderp/Llama-3.2-1B-Instruct-exl3 (4.0bpw branch) module by
module — the same harness shape as exllamav3's eval/model_diff.py — and
captures the residual stream after every TransformerBlock plus the final
logits. This is the Phase-4 oracle for tests/kld.uc: our ucode engine's
per-layer hidden states and logits are compared against it with
rel-RMS / cosine / KLD / perplexity metrics (NOT bit-exactness: the
reference graph runs fp32 intermediates while ours is fp16).

    fixtures/kld_oracle.safetensors :
        ids        int64   (S,)      input token ids (BOS first)
        layer{i}_h float16 (S, hid)  hidden state after block i, i = 0..15
        logits     float16 (S, V)    final logits for every position
        ppl_f32    float32 (1,)      oracle ppl: fp32 softmax over the
                                     stored fp16 logits, ids[1:] scored

Run from repo root (needs MODEL_DIR below + CUDA):
    .venv/bin/python tests/oracle/make_kld_oracle.py
"""
import os

import torch
from safetensors.torch import save_file
from exllamav3 import Model, Config
from exllamav3.tokenizer import Tokenizer

MODEL_DIR = "/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw"
NUM_TOKENS = 128

# Deterministic sample text; long enough to yield NUM_TOKENS pieces.
PROMPT = (
    "The history of computing is a history of abstraction. Each new "
    "layer hides the one beneath it: transistors became logic gates, "
    "gates became processors, processors became machines running "
    "operating systems, and operating systems now host interpreters "
    "that turn plain text into behaviour. A language model sits at the "
    "end of this chain. It reads text and predicts text, yet inside it "
    "runs the same old arithmetic: matrices multiplied, sums reduced, "
    "gradients once descended. The interesting question is never "
    "whether the arithmetic happens, but how much of it a given "
    "machine can afford, and how quickly. Every engine that serves a "
    "model is therefore an exercise in economy: the same graph, "
    "computed with fewer bytes, fewer kernels, and fewer wasted "
    "clocks, is a better engine. Quantization is the sharpest tool in "
    "that drawer: it trades a little precision for a lot of memory "
    "bandwidth, and memory bandwidth is what inference actually runs "
    "on. The art lies in knowing which bits carry meaning and which "
    "merely carry noise."
)

device = "cuda:0"
torch.cuda.set_device(0)

outdir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(outdir, exist_ok=True)

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
model.load()
tok = Tokenizer(config)

ids_t = tok.encode(PROMPT, add_bos = True)
ids = ids_t.reshape(-1).tolist()[:NUM_TOKENS]
assert len(ids) == NUM_TOKENS, f"need {NUM_TOKENS} tokens, got {len(ids)}"
assert ids[0] == tok.bos_token_id, f"first token should be BOS, got {ids[0]}"
print("first ids:", ids[:8])

# Load every module to the device (model_diff.py's per-module load).
config.stc.begin_deferred_load()
for module in model.modules:
    module.load(device)
config.stc.end_deferred_load()

input_ids = torch.tensor([ids], dtype=torch.long, device=device)

# Walk the module chain exactly like eval/model_diff.py does.
params = {}
state = model.prepare_inputs(input_ids, params)

first = model.first_block_idx
n_layers = config.num_hidden_layers

tensors = {"ids": torch.tensor(ids, dtype=torch.int64)}

for idx, module in enumerate(model.modules):
    state = module.forward(state, params)

    if first <= idx < first + n_layers:
        h = state[0].to(torch.float16).cpu().clone()
        print(f"block {idx - first:2d}: hidden {tuple(h.shape)}")
        tensors[f"layer{idx - first}_h"] = h

logits = state
if logits.dim() == 3:
    logits = logits[0]
print("logits:", tuple(logits.shape), logits.dtype)

# Oracle perplexity from the fp32 logits (same scoring convention as
# ucode/metrics.uc: position i scores ids[i + 1]).
logits_f32 = logits.float()
logprobs = torch.log_softmax(logits_f32, dim=-1)
targets = input_ids[0, 1:]
picked = logprobs[:-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
ppl = torch.exp(-picked.sum() / targets.numel()).item()
print(f"oracle ppl (fp32 logits, {targets.numel()} tokens): {ppl:.6f}")

tensors["logits"] = logits.to(torch.float16).cpu()
tensors["ppl_f32"] = torch.tensor([ppl], dtype=torch.float32)

path = os.path.join(outdir, "kld_oracle.safetensors")
save_file(tensors, path)
print("wrote", path)