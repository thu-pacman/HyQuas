#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
head=../build/logs/`date +%Y%m%d-%H%M%S`

name=$head-group
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -D COALESCE=0 2>&1 | tee $name/std.out
name1=$name

name=$head-blas
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DCOALESCE=1 2>&1 | tee $name/std.out
name2=$name

name=$head-mix
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DCOALESCE=2 2>&1 | tee $name/std.out
name3=$name

grep -r "Time Cost" $head-*/*.log