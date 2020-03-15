#!/bin/bash

if [ "$0" = "$BASH_SOURCE" -o  -z "$1" ]; then
	echo "Usage: source $0 REQUIRE_GPU_NUM"
	exit 1;
fi

REQUIRE_GPU_NUM=$1
echo Require GPU number: $REQUIRE_GPU_NUM

# get all line numbers (n + 2)
_FREE_GPUS=$(nvidia-smi --format=csv --query-gpu=memory.free | grep 32510 -n | cut -d ':' -f 1 | tr '\n' ' ')
# convert to real GPU numbers and normalize with xargs
FREE_GPUS=$(echo $(for i in $_FREE_GPUS; do echo $((i - 2)); done) | xargs)

if [ -z "$FREE_GPUS" ]; then
	echo "No free GPUs, wait for some time and try again!"
	return 1;
fi

FREE_GPU_NUM=$(((${#FREE_GPUS} + 1) / 2))
echo "Free GPUs: $FREE_GPU_NUM ($FREE_GPUS)"

if [ $FREE_GPU_NUM -lt $REQUIRE_GPU_NUM ]; then
	echo "Free GPUs not enough, wait for some time and try again!"
	return 1
fi

# truncate to get part of GPUs
USE_GPU=${FREE_GPUS:0:$(($REQUIRE_GPU_NUM * 2 - 1))}
echo Use GPU: $USE_GPU
export CUDA_VISIBLE_DEVICES="$USE_GPU"

