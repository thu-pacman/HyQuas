set -e
export CUDA_VISIBLE_DEVICES=0
export MPIRUN_CONFIG=""
name=../build/logs/`date +%Y%m%d-%H%M%S`

cd ../scripts

cp ../src/kernels/baseline.cu ../src/kernelOpt.cu

mkdir -p $name
./check_wrapper.sh $name -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DMICRO_BENCH=on -DTHREAD_DEP=9 2>&1 | tee $name/std.out
grep -r "Time Cost" $name/*.log | tee ../benchmark/logs/sharemem.log

cp ../src/kernels/swizzle.cu ../src/kernelOpt.cu
