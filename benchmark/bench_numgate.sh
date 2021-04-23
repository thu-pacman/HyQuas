#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
head=../build/logs/`date +%Y%m%d-%H%M%S`

cd ../scripts

cp ../src/kernels/baseline.cu ../src/kernelOpt.cu
source init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMICRO_BENCH=on -DTHREAD_DEP=9 2>&1
CUDA_VISIBLE_DEVICES=0 ./two-group-h | tee ../benchmark/logs/numgate-sm.log

cp ../src/kernels/swizzle.cu ../src/kernelOpt.cu
