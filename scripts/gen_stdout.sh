#!/bin/bash
source init.sh -DBACKEND=1 -DSHOW_SUMMARY=off
for test in ${tests[*]}; do
    echo $test
    ./main ../tests/input/$test.qasm > ../tests/output/$test.log
done
