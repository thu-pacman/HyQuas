#!/bin/bash
mkdir -p logs
mkdir -p logs/evaluator_v100

cd ../scripts
echo "OShareMem"
source ./init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMEASURE_STAGE=on -DLOG_EVALUATOR=on 2>&1
CUDA_VISIBLE_DEVICES=0 ../build/main ../tests/input/basis_change_28.qasm 2>&1 1>../benchmark/logs/evaluator_v100/OShareMem.log
cd ../benchmark

cd ../scripts
echo "TransMM MAT=5"
source ./init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMEASURE_STAGE=on -DLOG_EVALUATOR=on -DOVERLAP_MAT=off -DMAT=5 2>&1
CUDA_VISIBLE_DEVICES=0 ../build/main ../tests/input/basis_change_28.qasm 2>&1 1>../benchmark/logs/evaluator_v100/TransMM_5.log
cd ../benchmark

cd ../scripts
echo "TransMM MAT=6"
source ./init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMEASURE_STAGE=on -DLOG_EVALUATOR=on -DOVERLAP_MAT=off -DMAT=6 2>&1
CUDA_VISIBLE_DEVICES=0 ../build/main ../tests/input/basis_change_28.qasm 2>&1 1>../benchmark/logs/evaluator_v100/TransMM_6.log
cd ../benchmark

cd ../scripts
echo "TransMM MAT=7"
source ./init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off -DMEASURE_STAGE=on -DLOG_EVALUATOR=on -DOVERLAP_MAT=off -DMAT=7 2>&1
CUDA_VISIBLE_DEVICES=0 ../build/main ../tests/input/basis_change_28.qasm 2>&1 1>../benchmark/logs/evaluator_v100/TransMM_7.log
cd ../benchmark
