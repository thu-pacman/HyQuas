#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ulimit -s unlimited

source /opt/spack/share/spack/setup-env.sh
spack load cuda@11
NVPROF_COMMAND="nsys nvprof --profile-from-start=off -o test"
export MPIRUN_CONFIG=""
export tests_28="basis_change_28 bv_28 hidden_shift_28 qaoa_28 qft_28 quantum_volume_28 supremacy_28"
export tests="$tests_25 $tests_28 $tests_30"

head=../build/logs/`date +%Y%m%d-%H%M%S`
logdir=../benchmark/logs/
echo tests=$tests
cd ../scripts

name=$head-m3
mkdir -p $name
tests=$tests ./check_wrapper.sh $name -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMAT=3 -DMIN_MAT=3 2>&1 | tee $name/std.out
echo "+++++ 3" | tee $logdir/blas-profile.log 
for test in ${tests[*]}; do
    echo "===== $test" | tee -a $name/circ.profile
    $NVPROF_COMMAND ../build/main  ../tests/input/$test.qasm 2>&1 | tee tmp.profile
    grep "cutlass" tmp.profile | tee -a $name/circ.profile
    grep "void transpose" tmp.profile | tee -a $name/circ.profile
done
cat $name/circ.profile | tee -a $logdir/blas-profile.log
name3=$name

name=$head-m4
mkdir -p $name
tests=$tests ./check_wrapper.sh $name -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMAT=4 2>&1 | tee $name/std.out
echo "+++++ 4" | tee -a $logdir/blas-profile.log
for test in ${tests[*]}; do
    echo "===== $test" | tee -a $name/circ.profile
    $NVPROF_COMMAND ../build/main  ../tests/input/$test.qasm 2>&1 | tee tmp.profile
    grep "cutlass" tmp.profile | tee -a $name/circ.profile
    grep "void transpose" tmp.profile | tee -a $name/circ.profile
done
cat $name/circ.profile | tee -a $logdir/blas-profile.log
name4=$name

name=$head-m5
mkdir -p $name
tests=$tests ./check_wrapper.sh $name -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMAT=5 2>&1 | tee $name/std.out
echo "+++++ 5" | tee -a $logdir/blas-profile.log
for test in ${tests[*]}; do
    echo "===== $test" | tee -a $name/circ.profile
    $NVPROF_COMMAND ../build/main  ../tests/input/$test.qasm 2>&1 | tee tmp.profile
    grep "cutlass" tmp.profile | tee -a $name/circ.profile
    grep "void transpose" tmp.profile | tee -a $name/circ.profile
done
cat $name/circ.profile | tee -a $logdir/blas-profile.log
name5=$name

name=$head-m6
mkdir -p $name
tests=$tests ./check_wrapper.sh $name -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMAT=6 2>&1 | tee $name/std.out
echo "+++++ 6" | tee -a $logdir/blas-profile.log
for test in ${tests[*]}; do
    echo "===== $test" | tee -a $name/circ.profile
    $NVPROF_COMMAND ../build/main  ../tests/input/$test.qasm 2>&1 | tee tmp.profile
    grep "cutlass" tmp.profile | tee -a $name/circ.profile
    grep "void transpose" tmp.profile | tee -a $name/circ.profile
done
cat $name/circ.profile | tee -a $logdir/blas-profile.log
name6=$name

name=$head-m7
mkdir -p $name
tests=$tests ./check_wrapper.sh $name -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMAT=7 2>&1 | tee $name/std.out
echo "+++++ 7" | tee -a $logdir/blas-profile.log
for test in ${tests[*]}; do
    echo "===== $test" | tee -a $name/circ.profile
    $NVPROF_COMMAND ../build/main  ../tests/input/$test.qasm 2>&1 | tee tmp.profile
    grep "cutlass" tmp.profile | tee -a $name/circ.profile
    grep "void transpose" tmp.profile | tee -a $name/circ.profile
done
cat $name/circ.profile | tee -a $logdir/blas-profile.log
name7=$name

name=$head-m8
mkdir -p $name
tests=$tests ./check_wrapper.sh $name -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMAT=8 2>&1 | tee $name/std.out
echo "+++++ 8" | tee -a $logdir/blas-profile.log
for test in ${tests[*]}; do
    echo "===== $test" | tee -a $name/circ.profile
    $NVPROF_COMMAND ../build/main  ../tests/input/$test.qasm 2>&1 | tee tmp.profile
    grep "cutlass" tmp.profile | tee -a $name/circ.profile
    grep "void transpose" tmp.profile | tee -a $name/circ.profile
done
cat $name/circ.profile | tee -a $logdir/blas-profile.log
name8=$name

name=$head-m9
mkdir -p $name
tests=$tests ./check_wrapper.sh $name -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMAT=9 2>&1 | tee $name/std.out
echo "+++++ 9" | tee -a $logdir/blas-profile.log
for test in ${tests[*]}; do
    echo "===== $test" | tee -a $name/circ.profile
    $NVPROF_COMMAND ../build/main  ../tests/input/$test.qasm 2>&1 | tee tmp.profile
    grep "cutlass" tmp.profile | tee -a $name/circ.profile
    grep "void transpose" tmp.profile | tee -a $name/circ.profile
done
cat $name/circ.profile | tee -a $logdir/blas-profile.log
name9=$name

name=$head-m10
mkdir -p $name
tests=$tests ./check_wrapper.sh $name -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMAT=10 2>&1 | tee $name/std.out
echo "+++++ 10" | tee -a $logdir/blas-profile.log
for test in ${tests[*]}; do
    echo "===== $test" | tee -a $name/circ.profile
    $NVPROF_COMMAND ../build/main  ../tests/input/$test.qasm 2>&1 | tee tmp.profile
    grep "cutlass" tmp.profile | tee -a $name/circ.profile
    grep "void transpose" tmp.profile | tee -a $name/circ.profile
done
cat $name/circ.profile | tee -a $logdir/blas-profile.log
name10=$name

grep -r "Time Cost" $head-m*/*.log | tee $logdir/blas.log
grep -r "Total Groups" $head-*/*.log | tee -a $logdir/blas.log
