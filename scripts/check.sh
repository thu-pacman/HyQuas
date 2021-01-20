#!/bin/bash
name=../build/logs/`date +%Y%m%d-%H%M%S`
mkdir -p $name
./check_wrapper.sh $name -DBACKEND=blas -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on 2>&1 | tee $name/std.out
