#!/bin/sh
# Layer split over all GPUs (one GPU busy at a time, wavefront prefill).
# Simplest, best prefill per GPU memory; decode 23-25 ms/token, MTP 45-75 tok/s.
# Needs 5-6 GPUs of 24 GB for the 6.05 bpw checkpoint (GPUS=N chooses the count).
MODE=layer exec "$(dirname "$0")/serve.sh" "$@"
