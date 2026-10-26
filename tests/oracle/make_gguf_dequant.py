#!/usr/bin/env python3
"""Oracle for the GGUF reader and the ggml dequantization kernels (tests/gguf_reader.uc, tests/gguf_dequant.uc).

Writes, with the reference implementation of the `gguf` package (ggml's quantization formats in numpy):

  tests/fixtures/gguf_small.gguf          a small GGUF file: key/value metadata (every value type) and one tensor per
                                          ggml type the kernels handle (random but valid blocks, 24 rows each) plus a
                                          plain F32 / BF16 / I32 tensor and a 3-D IQ2_XXS tensor (an expert stack)
  tests/fixtures/gguf_small_ref.safetensors  the reference dequantization of every quantized tensor, float32

Needs: pip install gguf numpy safetensors   (test time only)
  python3 tests/oracle/make_gguf_dequant.py
"""
import os

import numpy as np
from gguf import GGMLQuantizationType as T, GGUFWriter, GGML_QUANT_SIZES, quants
from safetensors.numpy import save_file

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fixtures')
rng = np.random.default_rng(20261025)


def f16_bytes(n, lo=0.01, hi=0.5):
    return rng.uniform(lo, hi, n).astype(np.float16).view(np.uint8).reshape(n, 2)


def make_blocks(t, nblocks):
    """Random bytes with sane f16 scales (random scale fields would be NaN / inf half the time)."""
    blk, size = GGML_QUANT_SIZES[t]
    raw = rng.integers(0, 256, (nblocks, size), dtype=np.uint8)

    if t in (T.Q4_0, T.Q8_0, T.IQ4_NL, T.IQ2_XXS, T.IQ2_XS, T.IQ1_S, T.Q2_0, T.IQ2_S, T.IQ3_XXS, T.IQ3_S, T.IQ4_XS):
        raw[:, 0:2] = f16_bytes(nblocks)
    elif t in (T.Q5_K, T.Q4_K):
        raw[:, 0:2] = f16_bytes(nblocks)
        raw[:, 2:4] = f16_bytes(nblocks)
    elif t == T.Q3_K:
        raw[:, 108:110] = f16_bytes(nblocks)
    elif t == T.Q6_K:
        raw[:, 208:210] = f16_bytes(nblocks)
    elif t == T.IQ1_M:
        # the f16 super scale is spread over the top nibbles of the four u16 scales (bytes 49, 51, 53, 55 of the block)
        su = rng.uniform(0.01, 0.5, nblocks).astype(np.float16).view(np.uint16).astype(np.uint32)

        for k in range(4):
            raw[:, 49 + 2 * k] = (raw[:, 49 + 2 * k] & 0x0f) | (((su >> (4 * k)) & 0xf) << 4).astype(np.uint8)
    return raw


os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, 'gguf_small.gguf')
w = GGUFWriter(path, 'deepseek41')
w.add_uint32('test.u32', 4000000000)
w.add_int32('test.i32', -7)
w.add_uint8('test.u8', 200)
w.add_int64('test.i64', -(1 << 40))
w.add_float32('test.f32', 1.5)
w.add_float64('test.f64', 2.25)
w.add_bool('test.bool', True)
w.add_string('test.str', 'ampstein ü')
w.add_array('test.arr_i', [1, 2, 3])
w.add_array('test.arr_s', ['a', 'bc', ''])
w.add_array('test.arr_f', [0.5, 0.25])

refs = {}
for t in (T.Q4_0, T.Q8_0, T.IQ4_NL, T.Q5_K, T.IQ2_XXS, T.IQ2_XS, T.IQ1_M, T.IQ1_S,
          T.Q2_0, T.Q4_K, T.Q3_K, T.Q6_K, T.IQ2_S, T.IQ3_XXS, T.IQ3_S, T.IQ4_XS):
    blk, size = GGML_QUANT_SIZES[t]
    cols = 768
    rows = 24
    raw = make_blocks(t, rows * cols // blk).reshape(rows, cols // blk * size)
    ref = quants.dequantize(raw, t).astype(np.float32)
    assert ref.shape == (rows, cols) and np.isfinite(ref).all(), t.name
    name = t.name.lower()
    w.add_tensor(name, raw, raw_dtype=t)
    refs[name] = ref

# an expert stack: (experts, rows, cols) as ggml stores it (ne = cols, rows, experts)
raw = make_blocks(T.IQ2_XXS, 5 * 6 * 2).reshape(5, 6, 2 * 66)
refs['stack'] = quants.dequantize(raw, T.IQ2_XXS).astype(np.float32)
w.add_tensor('stack', raw, raw_dtype=T.IQ2_XXS)
w.add_tensor('plain_f32', rng.standard_normal((3, 8)).astype(np.float32))
w.add_tensor('plain_bf16', rng.integers(0, 65535, (4, 6), dtype=np.uint16).view(np.uint8).reshape(4, 12),
             raw_dtype=T.BF16)
w.add_tensor('plain_i32', np.arange(10, dtype=np.int32))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
save_file(refs, os.path.join(OUT, 'gguf_small_ref.safetensors'))
print('wrote', path, os.path.getsize(path), 'bytes')
