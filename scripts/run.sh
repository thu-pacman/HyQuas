#!/bin/bash
source ../scripts/init.sh -DUSE_GROUP=on -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 ./main ../tests/input/random_30.qasm