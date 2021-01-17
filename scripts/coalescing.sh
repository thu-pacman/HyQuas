#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

name=../build/logs/`date +%Y%m%d-%H%M%S`-c0
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -D COALESCE=0 2>&1 | tee $name/std.out
name1=$name

name=../build/logs/`date +%Y%m%d-%H%M%S`-c1
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DCOALESCE=1 2>&1 | tee $name/std.out
name2=$name

name=../build/logs/`date +%Y%m%d-%H%M%S`-c2
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DCOALESCE=2 2>&1 | tee $name/std.out
name3=$name

name=../build/logs/`date +%Y%m%d-%H%M%S`-c3
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DCOALESCE=3 2>&1 | tee $name/std.out
name4=$name

name=../build/logs/`date +%Y%m%d-%H%M%S`-c4
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DCOALESCE=4 2>&1 | tee $name/std.out
name5=$name

name=../build/logs/`date +%Y%m%d-%H%M%S`-c5
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on -DCOALESCE=5 2>&1 | tee $name/std.out
name6=$name

tail -n 9 $name1/std.out
tail -n 9 $name2/std.out
tail -n 9 $name3/std.out
tail -n 9 $name4/std.out
tail -n 9 $name5/std.out
tail -n 9 $name6/std.out
