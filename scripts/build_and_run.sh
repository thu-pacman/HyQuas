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
TASKS=$(ls answer | tr ' ' '\n' | xargs)
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
				echo "Skipped due to not enough GPUs"
				cd ../
				continue
			fi
		fi
		set -e
		/usr/bin/time -v ../../../$task
	    	for f in $(ls *); do
			ANSWER_PATH=../../../../tasks/answer/$task/$f
			if [ -f $ANSWER_PATH ]; then
				# if answer exists, compare with answer
				echo -ne "Checking $f against $ANSWER_PATH: "
				diff $ANSWER_PATH $f
				echo "PASS"
			elif [ $gpu -eq 1 ]; then
				# or generate temporary answer by 1 GPU
				echo -e "\nDetected $f with no answer present:"
				cat $f
			else
				# compare multiple-GPU result with 1 GPU
				echo -ne "Checking $f against ../1/$f: "
				diff ../1/$f $f
				echo "PASS"
			fi
		done
		cd ..
	done
	cd ..
done

