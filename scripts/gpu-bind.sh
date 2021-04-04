#!/bin/bash
rank=$OMPI_COMM_WORLD_LOCAL_RANK
GPU_start=$(( $rank * $GPUPerRank ))
GPU_end=$(( ($rank + 1) * $GPUPerRank - 1 ))
GPU=`echo $(for i in $(seq $GPU_start $GPU_end); do printf "$i,"; done)`
CUDA_VISIBLE_DEVICES=$GPU $@