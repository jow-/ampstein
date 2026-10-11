#!/usr/bin/env python3
"""Oracle-side benchmark: exllamav3 Generator decode throughput on the
same model + GPU the ucode engine is benchmarked on (examples/bench.uc).

NOT shipped — test/benchmark tooling only, same category as the other
tests/oracle scripts.

Run from repo root:
    .venv/bin/python tests/oracle/bench_exllamav3.py [model-dir] [prefill] [decode]
"""
import os
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                             "exllamav3"))

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw"
PREFILL = int(sys.argv[2]) if len(sys.argv) > 2 else 128
DECODE = int(sys.argv[3]) if len(sys.argv) > 3 else 64

torch.cuda.set_device(0)

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 2048, max_batch_size = 1)
model.load()
tokenizer = Tokenizer(config)

# Same deterministic token pattern as examples/bench.uc: ids 791 + (i % 16),
# BOS prepended by the generator's add_bos.
ids = [791 + (i % 16) for i in range(PREFILL)]

generator = Generator(model, cache, tokenizer)

# Warmup
job = Job(input_ids = torch.tensor([ids[:8]], dtype = torch.long), max_new_tokens = 4)
generator.enqueue(job)
while generator.num_remaining_jobs():
    for _ in generator.iterate():
        pass

torch.cuda.synchronize()

# Prefill + decode, timed as one generation (the number users see)
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

# Decode-only estimate: rerun with the same prompt length but time after
# the first iteration (prefill happens on the first iterate)
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