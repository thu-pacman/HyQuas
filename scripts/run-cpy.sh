#!/bin/bash
source ../scripts/init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on -DMICRO_BENCH=on -DUSE_DOUBLE=on -DDISABLE_ASSERT=off -DENABLE_OVERLAP=off -DMEASURE_STAGE=off -DEVALUATOR_PREPROCESS=on -DUSE_MPI=on -DMAT=7 -DREDUCE_BLOCK_DEP=6 -DMPI_GPU_GROUP_SIZE=4 -DINPLACE=10
`which mpirun` -host nico3:16 ../scripts/env.sh ../scripts/bind-cpy.sh ./main ../tests/input-old/basis_change_33.qasm
