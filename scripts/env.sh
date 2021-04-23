#!/bin/bash
case $(hostname -s) in
  nico*)
    echo "[CLUSTER] nico"
    source /opt/spack/share/spack/setup-env.sh
    spack load cuda@10.2.89 /v5oqq5n
    spack load openmpi@4.0.5 /h5eun6a
    spack load nccl@2.7.8-1 /l3466wl
    export NCCL_ROOT=/home/spack/opt/spack/linux-debian10-skylake_avx512/gcc-8.3.0/nccl-2.7.8-1-l3466wlxanfsfdna367pra5og2m7d3ut
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
    spack load cuda@10.2.89 /tlfcinz
    spack load openmpi@3.1.6 /5aaect6
    spack load nccl@2.7.8-1 /bapygpe
    export NCCL_ROOT=/opt/spack/opt/spack/linux-debian10-broadwell/gcc-8.3.0/nccl-2.7.8-1-bapygpevofy26tc7bl73enw73ntwvhmk
    ;;
  hanzo)
    echo "[CLUSTER] hanzo"
    source /opt/spack/share/spack/setup-env.sh
    export PATH=$HOME/package/cmake-3.19.2-Linux-x86_64/bin:/usr/mpi/gcc/openmpi-4.1.0rc5/bin:$PATH
    # use system mpi
    export CPATH=/usr/mpi/gcc/openmpi-4.1.0rc5/include:${CPATH-}
    spack load gcc@8.3.0 /liymwyb
    spack load cuda@10.2.89 /tlfcinz
    spack load nccl@2.7.8-1 /bapygpe
    export NCCL_ROOT=/opt/spack/opt/spack/linux-debian10-broadwell/gcc-8.3.0/nccl-2.7.8-1-bapygpevofy26tc7bl73enw73ntwvhmk
    ;;
  nova)
    echo "[CLUSTER] nova"
    source /opt/spack/share/spack/setup-env.sh
    spack load cuda@11 /njgeoec
    spack load openmpi /dfes7hw
esac

$@