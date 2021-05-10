#!/bin/bash
rank=$OMPI_COMM_WORLD_LOCAL_RANK
#!/bin/bash
export OMP_NUM_THREAD=4
case $rank in
    0|1|2|3)
        export socket=0
        export GPU=0
        ;;
    4|5|6|7)
        export socket=0
        export GPU=1
        ;;
    8|9|10|11)
        export socket=1
        export GPU=5
        ;;
    12|13|14|15)
        export socket=1
        export GPU=6
        ;;
esac
echo socket-binding:`hostname` $OMPI_COMM_WORLD_LOCAL_RANK s=$socket g=$GPU
CUDA_VISIBLE_DEVICES=$GPU numactl --cpunodebind=$socket -m $socket $@
