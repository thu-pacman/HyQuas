#!/bin/bash
set -u
set -e

source /usr/local/Modules/init/bash
module load cuda-10.2/cuda
module load cmake-3.12.3
cd $HOME/QCSimulator/build
rm CMakeCache.txt || true
cmake $* ..
make clean
make
export tests=(adder_26 basis_change_28 bv_28 hidden_shift_28 ising_25 qaoa_28 qft_28 quantum_volume_28 supremacy_28)