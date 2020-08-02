#!/bin/bash
source init.sh -DUSE_GROUP=off -DSHOW_SUMMARY=off
for test in ${tests[*]}; do
    echo $test
    ./main ../tests/input/$test.qasm > ../tests/output/$test.log
done
