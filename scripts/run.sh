#!/bin/bash
source ../scripts/init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on
CUDA_VISIBLE_DEVICES=0 ./main ../tests/input/qft_28.qasm