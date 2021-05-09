#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
head=../build/logs/`date +%Y%m%d-%H%M%S`

echo "Measure Time r=4" | tee ../benchmark/logs/measure-perf.log
source ../scripts/init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=on -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMAT=7 -DREDUCE_BLOCK_DEP=4
CUDA_VISIBLE_DEVICES=0 ./measure-perf 2>&1 | tee -a ../benchmark/logs/measure-perf.log

echo "Measure Time r=5" | tee -a ../benchmark/logs/measure-perf.log
source ../scripts/init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=on -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMAT=7 -DREDUCE_BLOCK_DEP=5
CUDA_VISIBLE_DEVICES=0 ./measure-perf 2>&1 | tee -a ../benchmark/logs/measure-perf.log

echo "Measure Time r=6" | tee -a ../benchmark/logs/measure-perf.log
source ../scripts/init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=on -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMAT=7 -DREDUCE_BLOCK_DEP=6
CUDA_VISIBLE_DEVICES=0 ./measure-perf 2>&1 | tee -a ../benchmark/logs/measure-perf.log

echo "Measure Time r=7" | tee -a ../benchmark/logs/measure-perf.log
source ../scripts/init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=on -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMAT=7 -DREDUCE_BLOCK_DEP=7
CUDA_VISIBLE_DEVICES=0 ./measure-perf 2>&1 | tee -a ../benchmark/logs/measure-perf.log

echo "Measure Time r=8" | tee -a ../benchmark/logs/measure-perf.log
source ../scripts/init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=on -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMAT=7 -DREDUCE_BLOCK_DEP=8
CUDA_VISIBLE_DEVICES=0 ./measure-perf 2>&1 | tee -a ../benchmark/logs/measure-perf.log

echo "Measure Time r=9" | tee -a ../benchmark/logs/measure-perf.log
source ../scripts/init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=on -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMAT=7 -DREDUCE_BLOCK_DEP=9
CUDA_VISIBLE_DEVICES=0 ./measure-perf 2>&1 | tee -a ../benchmark/logs/measure-perf.log

echo "Measure Time r=10" | tee -a ../benchmark/logs/measure-perf.log
source ../scripts/init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=on -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMAT=7 -DREDUCE_BLOCK_DEP=10
CUDA_VISIBLE_DEVICES=0 ./measure-perf 2>&1 | tee -a ../benchmark/logs/measure-perf.log

grep "Measure Time" ../benchmark/logs/measure-perf.log
