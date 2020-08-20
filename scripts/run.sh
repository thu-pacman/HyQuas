#!/bin/bash
source ../scripts/init.sh -DUSE_GROUP=on -DSHOW_SUMMARY=on -DSHOW_SCHEDULE=off -DMICRO_BENCH=on
./local-single 2>&1 | tee log3