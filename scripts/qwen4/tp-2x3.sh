#!/bin/sh
# Two pipelined TP groups of 3 GPUs (layers 0-23 / 24-47).
# Middle ground: decode 21.6 ms/token, prefill ~1.3k tok/s.
MODE=tp STAGES=2 GPUS=${GPUS:-6} exec "$(dirname "$0")/serve.sh" "$@"
