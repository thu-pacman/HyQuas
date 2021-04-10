#!/bin/bash
name=../build/logs/`date +%Y%m%d-%H%M%S`
mkdir -p $name
NUM_RANK=2 GPUPerRank=2 ./check_wrapper.sh $name -DBACKEND=mix -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=on -DENABLE_OVERLAP=off -DUSE_MPI=on -DDISABLE_ASSERT=on -DMAT=7 -DUSE_MPI=on 2>&1 | tee $name/std.out
