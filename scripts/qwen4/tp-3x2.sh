#!/bin/sh
# Three pipelined TP groups of 2 GPUs (layers 0-15 / 16-31 / 32-47). RECOMMENDED TP setting.
# Prefill ~2.0-2.1k tok/s (a wavefront over the groups), decode 23.2 ms/token,
# MTP 52-77 tok/s, ~18-19 GiB per GPU at ctx 131072.
MODE=tp STAGES=3 GPUS=${GPUS:-6} exec "$(dirname "$0")/serve.sh" "$@"
