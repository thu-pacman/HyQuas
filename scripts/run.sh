#!/bin/bash
source ../scripts/init.sh -DUSE_GROUP=on -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=on
mpirun -n 8 ./main ../tests/input/random_30.qasm