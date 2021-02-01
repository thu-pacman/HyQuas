#!/bin/bash
ulimit -s unlimited
PROFILE_CMD="nsys nvprof -o test"

source ../scripts/init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DMAT=3 -DMIN_MAT=3
CUDA_VISIBLE_DEVICES=0 ./bench-blas | tee blas-h.log
CUDA_VISIBLE_DEVICES=0 $PROFILE_CMD --profile-from-start=off ./bench-blas 2>&1 | grep 'void transpose' > blas-h.profile || true

source ../scripts/init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DMAT=4 -DMIN_MAT=4
CUDA_VISIBLE_DEVICES=0 ./bench-blas | tee -a blas-h.log
CUDA_VISIBLE_DEVICES=0 $PROFILE_CMD --profile-from-start=off ./bench-blas 2>&1 | grep 'void transpose' >> blas-h.profile || true

source ../scripts/init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DMAT=5 -DMIN_MAT=5
CUDA_VISIBLE_DEVICES=0 ./bench-blas | tee -a blas-h.log
CUDA_VISIBLE_DEVICES=0 $PROFILE_CMD --profile-from-start=off ./bench-blas 2>&1 | grep 'void transpose' >> blas-h.profile || true

source ../scripts/init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DMAT=6 -DMIN_MAT=6
CUDA_VISIBLE_DEVICES=0 ./bench-blas | tee -a blas-h.log
CUDA_VISIBLE_DEVICES=0 $PROFILE_CMD --profile-from-start=off ./bench-blas 2>&1 | grep 'void transpose' >> blas-h.profile || true

source ../scripts/init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DMAT=7 -DMIN_MAT=7
CUDA_VISIBLE_DEVICES=0 ./bench-blas | tee -a blas-h.log
CUDA_VISIBLE_DEVICES=0 $PROFILE_CMD --profile-from-start=off ./bench-blas 2>&1 | grep 'void transpose' >> blas-h.profile || true

source ../scripts/init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DMAT=8 -DMIN_MAT=8
CUDA_VISIBLE_DEVICES=0 ./bench-blas | tee -a blas-h.log
CUDA_VISIBLE_DEVICES=0 $PROFILE_CMD --profile-from-start=off ./bench-blas 2>&1 | grep 'void transpose' >> blas-h.profile || true

source ../scripts/init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DMAT=9 -DMIN_MAT=9
CUDA_VISIBLE_DEVICES=0 ./bench-blas | tee -a blas-h.log
CUDA_VISIBLE_DEVICES=0 $PROFILE_CMD --profile-from-start=off ./bench-blas 2>&1 | grep 'void transpose' >> blas-h.profile || true

source ../scripts/init.sh -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on -DUSE_DOUBLE=on -DMAT=10 -DMIN_MAT=10
CUDA_VISIBLE_DEVICES=0 ./bench-blas | tee -a blas-h.log
CUDA_VISIBLE_DEVICES=0 $PROFILE_CMD --profile-from-start=off ./bench-blas 2>&1 | grep 'void transpose' >> blas-h.profile || true