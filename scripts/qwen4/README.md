# Qwen3.8-Flash-Next server launchers

| script | layout | decode (plain) | prefill | notes |
|---|---|---|---|---|
| `layer-split.sh` | layer split | 23-25 ms/token | 1.6-2.7k tok/s | simplest |
| `tp6.sh` | TP-6 (1 group) | 20.8 ms/token | 0.5-0.7k tok/s | decode-heavy only |
| `tp-2x3.sh` | 2 groups x 3 GPUs | 21.6 ms/token | ~1.3k tok/s | |
| `tp-3x2.sh` | 3 groups x 2 GPUs | 23.2 ms/token | ~2.0k tok/s | recommended TP |

Measured on 6 x RTX A5000/A5500 (24 GB), 6.05 bpw checkpoint, MTP window 3 by default.
A 2-GPU tensor-parallel layout does not exist for this model: the weights are ~100 GB,
so it needs 5-6 cards of 24 GB. With 5 GPUs use `GPUS=5 ./layer-split.sh` (TP needs
`GPUS` divisible by the stage count: 6 = 1x6, 2x3, 3x2).

```sh
./tp-3x2.sh                          # defaults: port 8043, ctx 131072
PORT=8080 CTX=65536 ./tp-3x2.sh --no-vision
PREFIX=/opt/ampstein LOG=/tmp/qwen4.log ./layer-split.sh
```

Environment and options: see the header of `serve.sh`. `PREFIX` points at an
install prefix holding `bin/ampstein` (`cmake --install`; default: `build/stage`).
