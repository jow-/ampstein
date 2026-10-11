#!/bin/sh
# Common launcher for the Qwen3.8-Flash-Next OpenAI server. The layout wrappers
# (layer-split.sh, tp6.sh, tp-2x3.sh, tp-3x2.sh) set MODE / GPUS / STAGES and call this.
#
# Environment (all optional):
#   MODEL    checkpoint directory   (default /data/models/Qwen3.8-Flash-Next-exl3-6.05bpw)
#   PORT     listen port            (default 8043)
#   HOST     listen address         (default 127.0.0.1)
#   CTX      context length         (default 131072; ~19-21 GiB per GPU at 131072)
#   GPUS     number of GPUs         (default: all visible; the model needs 5-6 x 24 GB)
#   MODE     layer | tp             (default layer)
#   STAGES   TP groups              (tp only; GPUS must be divisible by it; default 1)
#   MTP      speculative window     (default 3; 0 disables)
#   CHUNK    prefill chunk          (default: server picks, 1024 / 4096 for tp)
#   PREFIX   install prefix holding bin/ampstein (default: <repo>/build/stage)
#   UCODE    ucode command  (default: ucode)
#   LD_EXTRA extra LD_LIBRARY_PATH entry (e.g. a local lib dir)
#   LOG      log file (default: stdout)
# Extra arguments are passed to the server (e.g. --no-vision, --cache-ram 64).

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

: "${MODEL:=/data/models/Qwen3.8-Flash-Next-exl3-6.05bpw}"
: "${PORT:=8043}"
: "${HOST:=127.0.0.1}"
: "${CTX:=131072}"
: "${MODE:=layer}"
: "${MTP:=3}"
: "${UCODE:=ucode}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -n "$LD_EXTRA" ] && export LD_LIBRARY_PATH="$LD_EXTRA:$LD_LIBRARY_PATH"

ARGS="--model $MODEL --host $HOST --port $PORT --ctx $CTX --mtp $MTP"
[ -n "$GPUS" ] && ARGS="$ARGS --gpus $GPUS"
[ -n "$CHUNK" ] && ARGS="$ARGS --chunk $CHUNK"

if [ "$MODE" = tp ]; then
	ARGS="$ARGS --tp --stages ${STAGES:-1}"
fi

echo "qwen4 server: mode=$MODE gpus=${GPUS:-all} stages=${STAGES:-1} ctx=$CTX mtp=$MTP port=$PORT" >&2

# The installed loader: <prefix>/bin/ampstein (cmake --install; it finds
# lib/ampstein/torch.so and share/ampstein/*.uc relative to itself). PREFIX
# defaults to the staging prefix used for development, see README "Building".
: "${PREFIX:=$ROOT/build/stage}"
ENTRY="$PREFIX/bin/ampstein"

# shellcheck disable=SC2086
if [ -n "$LOG" ]; then
	exec $UCODE "$ENTRY" $ARGS "$@" >"$LOG" 2>&1
fi

# shellcheck disable=SC2086
exec $UCODE "$ENTRY" $ARGS "$@"
