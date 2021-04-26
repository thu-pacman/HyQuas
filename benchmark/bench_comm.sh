#!/bin/bash
mkdir -p logs
mkdir -p logs/bench_comm
mkdir -p logs/bench_comm/4V100
mkdir -p logs/bench_comm/2V100
cd ../scripts
source ./init.sh -DBACKEND=mix -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off 2>&1
cd ../benchmark

tests="basis_change_28 bv_28 hidden_shift_28 qaoa_28 qft_28 quantum_volume_28 supremacy_28"

echo "test 4V100"

for test in $tests; do
echo $test
CUDA_VISIBLE_DEVICES=0,1,2,3 nvprof --print-gpu-trace ../build/main ../tests/input/$test.qasm 1>logs/bench_comm/4V100/$test.log 2>logs/bench_comm/4V100/$test.out
done

echo "test 2V100"

for test in $tests; do
echo $test
CUDA_VISIBLE_DEVICES=0,1 nvprof --print-gpu-trace ../build/main ../tests/input/$test.qasm 1>logs/bench_comm/2V100/$test.log 2>logs/bench_comm/2V100/$test.out
done
