#!/bin/bash
case $(hostname -s) in
  nico*)
    echo "[CLUSTER] nico"
    source /opt/spack/share/spack/setup-env.sh
    spack load cuda@10.2.89 /v5oqq5n
    spack load openmpi@4.0.5 /h5eun6a
    export NCCL_ROOT=/home/heheda/tools/nccl/build
    export CPATH=$NCCL_ROOT/include:$CPATH
    export LIBRARY_PATH=$NCCL_ROOT/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH
    ;;
  gorgon*)
    echo "[CLUSTER] gorgon"
    source /usr/local/Modules/init/bash
    module load cuda-10.2/cuda
    module load cmake-3.12.3
    module load openmpi-3.0.0
    ;;
  i*)
    echo "[CLUSTER] scc"
    source /opt/spack/share/spack/setup-env.sh
    spack load cuda@11.2.2 /jcseufw
    spack load openmpi@4.0.5 /s32f5ly
    export NCCL_ROOT=/home/heheda/software/nccl/build
    export CPATH=$NCCL_ROOT/include:$CPATH
    export LIBRARY_PATH=$NCCL_ROOT/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH
    ;;
  hanzo)
    echo "[CLUSTER] hanzo"
    source /opt/spack/share/spack/setup-env.sh
    export PATH=$HOME/package/cmake-3.19.2-Linux-x86_64/bin:/usr/mpi/gcc/openmpi-4.1.0rc5/bin:$PATH
    # use system mpi
    export CPATH=/usr/mpi/gcc/openmpi-4.1.0rc5/include:${CPATH-}
    spack load gcc@8.3.0 /liymwyb
    spack load cuda@10.2.89 /tlfcinz
    ;;
  nova)
    echo "[CLUSTER] nova"
    source /opt/spack/share/spack/setup-env.sh
    spack load cuda@11 /njgeoec
    spack load openmpi /dfes7hw
esac

$@