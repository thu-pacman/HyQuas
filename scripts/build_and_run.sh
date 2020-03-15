#!/bin/bash

set -e

# ensure GPUs
#GPU_NUMS="1 2 4 8"
GPU_NUMS="1"

# load CUDA with spack
source /opt/spack/share/spack/setup-env.sh
spack load cuda@10.2.89

# detect tasks
cd tasks
TASKS=$(ls *.cpp | tr ' ' '\n' | cut -d '.' -f 1 | xargs)
echo "Tasks to run: $TASKS"
cd ..

# build project
rm -rf build && mkdir -p build && cd build
cmake ..
make -j

mkdir -p run && cd run

# run each tsk
for task in $TASKS; do
	mkdir -p $task && cd $task
	echo -e "\n\n=====Running task $task====="
	for gpu in $GPU_NUMS; do
		echo -e "\n---Trying to run test on $gpu GPUs---"
		mkdir -p $gpu && cd $gpu
		set +e
		source ../../../../scripts/require_gpus.sh $gpu
		if [ $? -ne 0 ]; then
			if [ $gpu -eq 1 ]; then
				echo "Not even one GPU usable, abort."
				exit 1
			else
				echo "Skip due to not enough GPUs"
				continue
			fi
		fi
		set -e
		/usr/bin/time -v ../../../$task
	    	for f in $(ls *); do
			if [ $gpu -eq 1 ]; then
				echo -e "\nOutput file: $f"
				cat $f
			else
				diff $f ../1/$f
			fi
		done
		cd ..
	done
	cd ..
done

