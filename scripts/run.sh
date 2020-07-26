#!/bin/bash
set -u
set -e

source /usr/local/Modules/init/bash
module load cuda-10.2/cuda
module load cmake-3.12.3
cd $HOME/QCSimulator/build
cmake ..
make clean
make
./parser $HOME/QC/qc-benchmark/output/adder_26
