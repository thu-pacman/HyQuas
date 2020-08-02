#!/bin/bash
task=supremacy_28
source ../scripts/init.sh -DUSE_GROUP=off -DSHOW_SUMMARY=off -DSHOW_SCHEDULE=off
./main ../tests/input/$task.qasm 2>&1 | tee log1
source ../scripts/init.sh -DUSE_GROUP=on -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=on
./main ../tests/input/$task.qasm 2>&1 | tee log2
diff log1 log2