#!/bin/bash
CUR_PATH=`pwd`

cp ../src/kernels/baseline.cu ../src/kernelOpt.cu
source init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMICRO_BENCH=on -DTHREAD_DEP=9 2>&1
echo "baseline" | tee pergate.log
CUDA_VISIBLE_DEVICES=0 ./local-single 2>&1 | tee -a pergate.log

cd $CUR_PATH
cp ../src/kernels/baseline.cu ../src/kernelOpt.cu
source init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMICRO_BENCH=on 2>&1
echo "multitask" | tee -a pergate.log
CUDA_VISIBLE_DEVICES=0 ./local-single 2>&1 | tee -a pergate.log

cd $CUR_PATH
cp ../src/kernels/lookup.cu ../src/kernelOpt.cu
source init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMICRO_BENCH=on 2>&1
echo "lookup" | tee -a pergate.log
CUDA_VISIBLE_DEVICES=0 ./local-single 2>&1 | tee -a pergate.log

cd $CUR_PATH
cp ../src/kernels/swizzle.cu ../src/kernelOpt.cu
source init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMICRO_BENCH=on 2>&1
echo "bank" | tee -a pergate.log
CUDA_VISIBLE_DEVICES=0 ./local-single 2>&1 | tee -a pergate.log

cat pergate.log
