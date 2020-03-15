#!/bin/bash

set -e

# ensure GPUs
REQUIRE_GPU_NUMS=1
source scripts/require_gpus.sh $REQUIRE_GPU_NUMS

# load CUDA with spack
source /opt/spack/share/spack/setup-env.sh
spack load cuda@10.2.89

# detect tasks
cd tasks
TASKS=$(ls *.cpp | tr ' ' '\n' | cut -d '.' -f 1 | xargs)
echo "Tasks to run: $TASKS"
cd ..

# build project
mkdir -p build && cd build
cmake ..
make -j

mkdir -p run && cd run

# run each tsk
for task in $TASKS; do
	mkdir -p $task && cd $task
	echo -e "\n\n=====Running task $task====="
	/usr/bin/time -v ../../$task
	for f in $(ls *); do
		echo -e "\nOutput file: $f"
		cat $f
	done
	cd ..
done

