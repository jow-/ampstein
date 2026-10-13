# DFlash 2 block drafter (Qwen3.5 / 3.8, tensor parallel)

Status: implemented for the Qwen3.8-27B family on `--tp` (`dflash.uc`,
`qwen35_tp.uc`, engine hooks). Companion to `PARALLEL.md`.

DFlash 2 ([z-lab/dflash](https://github.com/z-lab/dflash), Inco AI's
`Qwen3.8-27B-DFlash2`) is a block-diffusion drafter: one forward over the
anchor token plus 7 mask tokens proposes 7 tokens at once, conditioned on
hidden states of the target. It replaces the sequential MTP draft chain;
the verify window, GDN state rollback and the acceptance rules are the
engine's existing speculative round, unchanged. Drafts are a deterministic
(greedy) proposal, so greedy and sampled output stay lossless (the one-hot
draft rule of `mtp_round`).

## Usage

```sh
ampstein --model /data/models/Qwen3.8-27B-exl3-5.00bpw --tp --ctx 262144 \
         --dflash /data/models/Qwen3.8-27B-DFlash2-EXL3-4.00bpw
```

`--mtp N` is the number of drafts per round (default 7 with `--dflash`, the
drafter's block size is 8). The MTP head is not loaded in this mode. The
drafter needs the captured verify graph (it is built after it). It is
tensor parallel like the target (below): ~0.5 GB of weights and ring per
rank, plus ~0.25 GB of selector codebooks on rank 0.

## Model

Per round, for the block rows `[anchor, mask x W]` at positions
`start .. start + W`:

- **Context.** The target's residual stream after layers
  `target_layer_ids` (5, 19, 33, 47, 61) of the rows of the last verify
  (or the prompt) is concatenated (5 x 5120), projected by `fc` and
  RMS-normed (`hidden_norm`). Every draft layer turns it into extra
  keys / values with its own `k_proj` / `v_proj` (+ `k_norm`, RoPE at the
  rows' positions), kept in a per-layer ring of 4096 slots.
- **Layers (5).** Qwen3 blocks: `input_layernorm`, a two-tap grouped
  dynamic causal convolution, attention, the second convolution,
  residual; then `post_attention_layernorm`, convolution, SwiGLU MLP,
  convolution, residual. Attention is non-causal inside the block and
  sliding-window (2048) over the context. The block's q rows are the only
  queries.
- **Convolution.** `out[t] = (base[0][c] + d[t][0][c / 16]) * x[t] +
  (base[0][..tap 1] + d[t][1][c / 16]) * x[t - 1]` with `x[-1] = 0` (taps
  never cross the block start); `d` comes from `kernel_projection(x)` of
  the sublayer's *input*, the "post" kernel (`base[1]`, `d[..][1]`) is
  applied to the sublayer's output.
- **Head and selector.** The final-norm hidden of rows 1..W is scored by
  the TARGET's `lm_head` (the vocabulary-sharded heads of the TP model);
  the top 16 per row are kept (per shard, then merged), and the selector
  walks a path: `score(k) = unary[p][k] + <pred_cb[prev] * hp(h_p),
  succ_cb[cand[p][k]]>` with `hp = hidden_projection`, `prev` the anchor
  for p = 0 and else the token chosen at p - 1; the greedy choice is the
  argmax (`amp_dflash_select`, one block).

## Tensor parallel

The drafter is sharded over the target's ranks with the target's shares
(`tp.plan.weights`, `qwen35_tp.split_n`): attention by kv-head groups (q
follows, each rank keeps the ring of its heads), the MLP by 256-column
ranges, `o_proj` / `down_proj` / `fc` by input rows with an all-reduce of
the fp16 partials (11 per round). The fp32 residual stream, the norms and
the convolutions' kernel projections are replicated; each rank scores its
own copy of the final hidden with its vocabulary shard of the head, and
the top-k merge and the selector run on rank 0. On rank 0 alone the
drafter's ~4.7 ms per round left rank 1 waiting in the next all-reduce;
sharded, a round is ~1-1.8 ms shorter (L133).

## Engine integration

- `qwen35_tp.set_taps(ids)` records the residual stream after the tap
  layers, every rank's copy: `forward()` returns `feat` (per tap, per
  rank) and the captured verify graphs write persistent per-tap buffers
  (`graph.feat(S)`); `prefill_pipeline` passes every chunk's taps to
  `opts.on_taps`.
- `dflash.prefill_rows(feats, pos)` writes the newest ~2100 rows of a
  chunk into the ring (the engine skips chunks before the last window).
- `mtp_round` calls `DF.draft_async` (a CUDA graph, ids land in the round
  buffer on the device) and continues in the existing async greedy path:
  device-side acceptance, one wake-up per round. Sampled requests take the
  synchronous path with the same drafts.
- A round's context rows are the previous verify's W rows: only `a + 1`
  are real, the rest sit at positions >= the new block start and are
  overwritten by the block's own rows in the same pass (the visibility
  rule is `|q - k| < window` and `k <= block end`, and the slot written).
- Cache jumps (snapshot restore, fresh prefill, parked-state resume) reset
  the ring (`DF.reset()`); the context then refills from the next rounds'
  rows. Output stays lossless either way, only acceptance suffers.
- The verify window of 7 rows is 42 query rows per kv head (rep 6):
  `attn_decode`'s 48-row mma tile takes them in one pass over the keys
  (L132).

## Numerics: the residual stream is fp32

The drafter's attention / MLP outputs carry massive-activation channels
(~5e3 on channel 3629 from layer 0) that overflow fp16 after one or two
layers; its reference runs in bf16. The stream is kept in fp32 here
(`rms_norm_f32`, `dflash_conv_res` adds `scale * conv(y)` in place), and
`o_proj` / `down_proj` run with their `svh` pre-scaled by 1/16 so the fp16
products stay in range (the add scales back). Everything else is fp16.

## Measured (2 x RTX 3090, 220 W, Qwen3.8-27B 5.0 bpw, greedy, 400 tokens)

| prompt | MTP window 3 | DFlash window 7 |
|---|---|---|
| code (LRU cache) | 93 tok/s | 158 tok/s (5.1 tokens/round) |
| prose (essay) | 80 | 98 (3.2) |
| list (primes) | 102 | 214 (6.9) |
| summary at 26k depth | 62 | 69-73 (2.8-2.9) |

With the one-pass verify attention (L132) and the sharded drafter (L133),
at depth (the repository's docs and sources as of 9e10afd cut to the
depth, then a code / a summary request, 300 tokens):

| depth | code | prose | round |
|---|---|---|---|
| 1k | 152 tok/s (140 before) | 106 (99) | 30.5 ms (32.4) |
| 26k | 140 (116) | 100 (82) | 32.6 ms (39.9) |
| 82k | 127 (102) | 70 (55) | 36.3 ms (46.8) |
| 199k | 96 | 61 | 44.5 ms |

A round is ~26 ms of verify (7 rows cost about what 3 do), ~5.5 ms of
drafter (5 layers of 4-bit weights at m = 8 and the sharded head) and no
host gap. Windows 4 / 5 / 6 measured 120 / 137 / 146 tok/s on code.

## Tests

`tests/dflash_draft.uc` prefills a chat prompt with taps, drafts one
block and requires >= 4 of 7 drafts to equal the target's greedy
continuation. `tests/bindings_errors.uc` covers the four kernels
(`dflash_conv`, `dflash_conv_res`, `rms_norm_f32`, `dflash_select`).
`AMP_DFLASH_TRACE=1` prints per-stage tensor magnitudes of every round.
