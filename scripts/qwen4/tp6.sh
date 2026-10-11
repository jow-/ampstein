#!/bin/sh
# One tensor/expert-parallel group over all 6 GPUs (TP-6).
# Fastest decode (20.8 ms/token, MTP 57-84 tok/s) but host-bound prefill
# (~0.5-0.7k tok/s) and the least memory headroom (~21 GiB per GPU at ctx 131072).
# Use for decode-heavy sessions with a warm prompt cache.
MODE=tp STAGES=1 GPUS=${GPUS:-6} exec "$(dirname "$0")/serve.sh" "$@"
