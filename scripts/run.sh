#!/bin/bash
source ../scripts/init.sh -DBACKEND=mix -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=on -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off
GPUPerRank=2 `which mpirun` -n 2 ../scripts/gpu-bind.sh ./main ../tests/input-old/h_28.qasm
