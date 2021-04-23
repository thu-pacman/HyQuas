#!/bin/bash
set -u
set -e

source env.sh ""

cd $HOME/QCSimulator/build
rm CMakeCache.txt || true
cmake $* ..
make clean
make -j

if [ -z "${tests-}" ]
then
  export tests_25="basis_change_25 bv_25 hidden_shift_25 qaoa_25 qft_25 quantum_volume_25 supremacy_25"
  export tests_28="basis_change_28 bv_28 hidden_shift_28 qaoa_28 qft_28 quantum_volume_28 supremacy_28"
  export tests_30="basis_change_30 bv_30 hidden_shift_30 qaoa_30 qft_30 quantum_volume_30 supremacy_30"
  export tests_scale="basis_change_24 basis_change_25 basis_change_26 basis_change_27 basis_change_28"

  export tests=($tests_28)
fi
