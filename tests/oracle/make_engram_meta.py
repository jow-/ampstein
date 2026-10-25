#!/usr/bin/env python3
"""Model preparation for DeepSeek-V4.1 Engram: derive the hash tables ampstein needs from the
checkpoint's tokenizer.json and config.json, exactly as DeepSeek's inference/engram.py does, and
write them to <model_dir>/engram_meta.safetensors:

  token_map   int32 (vocab,)          token id -> compressed id (normalizer collapse)
  multipliers int64 (layers, ngram)   per-layer odd hash multipliers (numpy default_rng(10007 * layer))
  primes      int64 (layers, ngram-1, heads)  bucket moduli
  offsets     int64 (layers, (ngram-1)*heads) bucket range starts inside a layer's table
  pad         int64 (1,)              compressed id of engram_pad_token_id

If the model directory ships engram_token_map.u32 (little-endian uint32 per token id, as the 1017 packs do) it is used
instead of deriving the map from tokenizer.json, and `tokenizers` is not needed.

Needs: numpy, sympy, safetensors (+ tokenizers without engram_token_map.u32); not needed at serving time; run once per model.
  python3 make_engram_meta.py /path/to/model-dir
"""
import json
import os
import sys

import numpy as np
from safetensors.numpy import save_file
from sympy import isprime
try:
    from tokenizers import Regex, Tokenizer, normalizers
except ImportError:
    Tokenizer = None


def find_next_prime(start, seen):
    c = start + 1
    while not isprime(c) or c in seen:
        c += 1
    return c


def build_token_map(backend, n):
    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    key_to_new, lookup = {}, [0] * n
    for tid in range(n):
        text = backend.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(tid)
        else:
            t = norm.normalize_str(text)
            key = t if t else text
        nid = key_to_new.get(key)
        if nid is None:
            nid = len(key_to_new)
            key_to_new[key] = nid
        lookup[tid] = nid
    return lookup, len(key_to_new)


def main(model_dir):
    cfg = json.load(open(f"{model_dir}/config.json"))
    tc = cfg.get("text_config", cfg)
    layers = tc["engram_layer_ids"]
    nmax, heads, vs = tc["engram_max_ngram_size"], tc["engram_n_heads"], tc["engram_vocab_size"]

    n = tc["vocab_size"]
    shipped = f"{model_dir}/engram_token_map.u32"
    if os.path.exists(shipped):
        token_map = np.fromfile(shipped, dtype="<u4").astype(np.int64).tolist()
        assert len(token_map) == n, (len(token_map), n)
        cvs = max(token_map) + 1
    else:
        backend = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        token_map, cvs = build_token_map(backend, n)
    assert cvs == tc["engram_compressed_vocab_size"], (cvs, tc["engram_compressed_vocab_size"])

    primes, seen = [], set()
    for _ in layers:
        per = []
        for _ in range(nmax - 1):
            cur, sizes = vs - 1, []
            for _ in range(heads):
                cur = find_next_prime(cur, seen)
                seen.add(cur)
                sizes.append(cur)
            per.append(sizes)
        primes.append(per)

    offsets = []
    for per in primes:
        flat = [p for sizes in per for p in sizes]
        offsets.append(np.cumsum([0, *flat[:-1]]))

    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // cvs) // 2)
    mult = []
    for lid in layers:
        g = np.random.default_rng(10007 * lid)
        v = g.integers(low=0, high=bound, size=(nmax,), dtype=np.int64)
        mult.append(v * 2 + 1)

    out = {
        "token_map": np.array(token_map, dtype=np.int32),
        "multipliers": np.array(mult, dtype=np.int64),
        "primes": np.array(primes, dtype=np.int64),
        "offsets": np.array(offsets, dtype=np.int64),
        "pad": np.array([token_map[tc["engram_pad_token_id"]]], dtype=np.int64),
    }
    save_file(out, f"{model_dir}/engram_meta.safetensors")
    print("wrote", f"{model_dir}/engram_meta.safetensors", {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main(sys.argv[1])
