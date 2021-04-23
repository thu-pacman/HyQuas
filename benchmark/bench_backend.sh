set -e
export CUDA_VISIBLE_DEVICES=0
export MPIRUN_CONFIG=""
head=../build/logs/`date +%Y%m%d-%H%M%S`

cd ../scripts

name=$head-group
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DUSE_MPI=off 2>&1 | tee $name/std.out
grep -r "Time Cost" $name/*.log | tee ../benchmark/logs/backend.log

name=$head-blas
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DUSE_MPI=off 2>&1 | tee $name/std.out
grep -r "Time Cost" $name/*.log | tee -a ../benchmark/logs/backend.log

name=$head-mix
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=mix -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DEVALUATOR_PREPROCESS=on -DUSE_MPI=off 2>&1 | tee $name/std.out
grep -r "Time Cost" $name/*.log | tee -a ../benchmark/logs/backend.log

