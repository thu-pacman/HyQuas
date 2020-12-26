#!/bin/bash
source ../scripts/init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off
CUDA_VISIBLE_DEVICES=0 ./main ../tests/input/qft_28.qasm