#!/bin/bash
name=../build/logs/`date +%Y%m%d-%H%M%S`
mkdir -p $name

# command for no_mpi
MPIRUN_CONFIG="" ./check_wrapper.sh $name -DBACKEND=mix -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=on -DENABLE_OVERLAP=on -DUSE_MPI=off -DDISABLE_ASSERT=on -DMAT=7 2>&1 | tee $name/std.out

# command for mpi
MPIRUN_CONFIG="`which mpirun` -x GPUPerRank=2 -host nico3:2 ../scripts/env.sh ../scripts/gpu-bind.sh"
MPIRUN_CONFIG=$MPIRUN_CONFIG ./check_wrapper.sh $name -DBACKEND=mix -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=on -DENABLE_OVERLAP=on -DUSE_MPI=on -DDISABLE_ASSERT=on -DMAT=7 -DUSE_MPI=on 2>&1 | tee $name/std.out
