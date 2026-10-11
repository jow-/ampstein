#!/usr/bin/env python3
"""Oracle-side benchmark: exllamav3 Generator decode throughput on the
Qwen3.8-27B exl3 model, autosplit over the same GPUs the ucode engine
uses (examples/bench_qwen35.uc — layers 0-31 / 32-63 over cuda:0/1).

NOT shipped — test/benchmark tooling only.

Run from repo root:
    .venv/bin/python tests/oracle/bench_qwen35_exllamav3.py [model-dir] [prefill] [decode]
"""
import os
import sys
import time

os.environ.setdefault("EXLLAMAV3_TUNE_CACHE", "/tmp/exl3_tune_qwen35")

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                             "exllamav3"))

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/models/Qwen3.8-27B-exl3-5.00bpw"
PREFILL = int(sys.argv[2]) if len(sys.argv) > 2 else 128
DECODE = int(sys.argv[3]) if len(sys.argv) > 3 else 64

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 2048, max_batch_size = 1)
model.load()
tokenizer = Tokenizer(config)

# Same deterministic token pattern as examples/bench_qwen35.uc.
ids = [791 + (i % 16) for i in range(PREFILL)]

generator = Generator(model, cache, tokenizer)

# Warmup
job = Job(input_ids = torch.tensor([ids[:8]], dtype = torch.long), max_new_tokens = 4)
generator.enqueue(job)
while generator.num_remaining_jobs():
    for _ in generator.iterate():
        pass

torch.cuda.synchronize()

# Prefill + decode, timed as one generation
t0 = time.perf_counter()
job = Job(input_ids = torch.tensor([ids], dtype = torch.long), max_new_tokens = DECODE)
generator.enqueue(job)

n_new = 0
while generator.num_remaining_jobs():
    for res in generator.iterate():
        n_new += len(res.get("token_ids") or [])
torch.cuda.synchronize()
t1 = time.perf_counter()

total = t1 - t0
print(f"exllamav3: {n_new} new tokens in {total * 1000:.1f} ms "
      f"-> {n_new / total:.2f} tok/s (incl. {PREFILL}-token prefill)")

# Decode-only estimate
job = Job(input_ids = torch.tensor([ids], dtype = torch.long), max_new_tokens = DECODE)
generator.enqueue(job)
first = True
t0 = time.perf_counter()
n_new = 0
while generator.num_remaining_jobs():
    for res in generator.iterate():
        if first:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            first = False
        n_new += len(res.get("token_ids") or [])
torch.cuda.synchronize()
t1 = time.perf_counter()
if n_new:
    dt = t1 - t0
    print(f"exllamav3 decode-only: {n_new} tokens in {dt * 1000:.1f} ms "
          f"-> {n_new / dt:.2f} tok/s")