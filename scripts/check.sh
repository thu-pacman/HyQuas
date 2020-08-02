#!/bin/bash
name=../build/logs/`date +%Y%m%d-%H%M%S`
mkdir -p $name
./check_wrapper.sh $name 2>&1 | tee $name/stdout.log
