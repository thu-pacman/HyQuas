#!/bin/bash
set -x
source init.sh -DBACKEND=group -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DUSE_DOUBLE=on
input_dir=../tests/input
std_dir=../tests/output

for test in ${tests[*]}; do
    ./main $input_dir/$test.qasm > $1/$test.log
    grep "Logger" $1/$test.log
done

set +x
set +e

for test in ${tests[*]}; do
    line=`cat $std_dir/$test.log | wc -l`
    echo $test
    diff -B <(head -n $line $std_dir/$test.log) <(head -n $line $1/$test.log) || true
done

grep -r "Time Cost" $1/*.log 