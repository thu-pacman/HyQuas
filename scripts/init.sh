#!/bin/bash
set -u
set -e

case $(hostname -s) in
  nico*)
    echo "nico cluster"
    source /opt/spack/share/spack/setup-env.sh
    spack load cuda@10.2.89 /v5oqq5n
    spack load openmpi@4.0.4 /24my3rr
    ;;
  gorgon*)
    echo "gorgon cluster"
    source /usr/local/Modules/init/bash
    module load cuda-10.2/cuda
    module load cmake-3.12.3
    module load openmpi-3.0.0
    ;;
  i*)
    echo "scc cluster"
    source /opt/spack/share/spack/setup-env.sh
    spack load cuda@10.2.89 /odirgft
    spack load openmpi@3.1.4 /cmuktug
    ;;
  hanzo)
    echo "hanzo cluster"
    source /opt/spack/share/spack/setup-env.sh
    export PATH=$HOME/package/cmake-3.19.2-Linux-x86_64/bin:$PATH
    spack load cuda@10.2.89 /odirgft
    spack load openmpi@3.1.6
    rsync -avz nico4:~/QCSimulator/src $HOME/QCSimulator
    rsync -avz nico4:~/QCSimulator/micro-benchmark $HOME/QCSimulator 
    ;;
  nova)
    echo "nova cluster"
    source /opt/spack/share/spack/setup-env.sh
    spack load cuda@11 /njgeoec
    spack load openmpi /dfes7hw
esac

cd $HOME/QCSimulator/build
rm CMakeCache.txt || true
cmake $* ..
make clean
make -j
export tests=(adder_26 basis_change_28 bv_28 hidden_shift_28 ising_25 qaoa_28 qft_28 quantum_volume_28 supremacy_28)
