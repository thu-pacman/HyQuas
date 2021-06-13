# HyQuas

HyQuas is a **Hy**brid partitioner based **Qua**ntum circuit **S**imulation system on GPU, which supports both single-GPU, single-node-multi-GPU, and multi-node-multi-GPU quantum circuit simulation.

For single-GPU simulation, it provides two highly optimized methods, *OShareMem* and *TransMM*. *OShareMem* method optimizes the shared-memory based quantum circuit simulation by <img src="https://render.githubusercontent.com/render/math?math=2.67 \times">. *TransMM* method converts quantum circuit simulation to standard operations and enables the usage of highly optimized libraries like cuBLAS and powerful hardwares like Tensor Cores. It leads up to <img src="https://render.githubusercontent.com/render/math?math=8.43 \times"> speedup over previous gate-merging based simulation. Moreover, it can select the better simulation method for different parts of a given quantum circuit according to its pattern. 

For distributed simulation, it provides a GPU-centric communication pipelining approach. It can utilize the high-throughput NVLink connections to make the simulation even faster while still preserving low communication traffic.

Experimental results show that HyQuas can achieve up to <img src="https://render.githubusercontent.com/render/math?math=10.71 \times"> speedup on a single GPU and <img src="https://render.githubusercontent.com/render/math?math=227 \times"> speedup on a GPU cluster over state-of-the-art quantum circuit simulation systems.

For more details, please refer to our paper:
```
Chen Zhang, Zeyu Song, Haojie Wang, Kaiyuan Rong, and Jidong Zhai. 2021. HyQuas: hybrid partitioner based quantum circuit simulation system on GPU. In Proceedings of the ACM International Conference on Supercomputing (ICS '21). Association for Computing Machinery, New York, NY, USA, 443–454. DOI:https://doi.org/10.1145/3447818.3460357
```

## Compile and Run
1. Get the source code
    ```bash
    git clone https://github.com/thu-pacman/HyQuas.git --recursive
    ```

2. Specify the compute capability in `CMakeLists.txt` (`CUDA_NVCC_FLAGS`) and `third-party/cutt/Makefile` (`GENCODE_FLAGS`)

3. Prepare the following dependencies
    * cmake (tested on 3.12.3)
    * cuda (tested on 10.2.89 and 11.0.2)
    * g++ (compatible with cuda)
    * cublas (with the same version of cuda)
    * openmpi (tested on 4.0.5)
    * nccl (Fully tested on 2.9.6-1. Known that 2.7.8-1 cannot work. It will be blocked in an NCCL simulated MPI_Sendrecv.)
    And update environment variables like `CUDA_HOME`, `NCCL_ROOT`, `$PATH`, `$LIBRARY_PATH`, `$LD_LIBRARY_PATH`, `CPATH` in `scripts/env.sh`.

4. Compile the tensor transpose library `cutt`

    ```bash
    cd third-party/cutt
    make -j
    ```

5. Specify the root directory
    ```bash
    export HYQUAS_ROOT=${The_directory_running_git_clone}/HyQuas
    ```

5. Prepare the database for the time predictor
    ```bash
    mkdir -p evaluator-preprocess/parameter-files
    cd benchmark
    ./preprocess.sh
    ```

6. Example usages of HyQuas:
    HyQuas will use all GPUs it can detect, so please control the number of GPU by `CUDA_VISIBLE_DEVICES`.
    * Run a single circuit with single GPU
        ```bash
        cd scripts
        ./run-single.sh
        ```

    * Run a single circuit with multiple GPUs in one node
        ```bash
        cd scripts
        ./run-multi-GPU.sh
        ```

    * Run a single circuit with multiple GPUs in multiple nodes
        Please modify the `-host` first.
        ```bash
        cd scripts
        ./run-multi-node.sh
        ```

    * Run all circuits and check the correctness (The script trys both w/o MPI)
        ```bash
        cd scripts
        CUDA_VISIBLE_DEVICES=0,1,2,3 ./check.sh
        ```

For more ways to use our simulator (like only using the *OShareMem* method or *TransMM* method, tuning off the overlap of communication and computation), and for reproducing our results in the ICS'21 paper, please refer to our `benchmark/` directory.

The HyQuas system is still under development. It has the following **unstable** feathers now. See our dev branch for details.
* Simulating more qubits by saving the state in CPU memory while still compute with GPU.
* An imperative mode, so that you do not need to explicitly call `c->compile();` and `c->run()`.
* Support for more control qubits.
* Support for some two-qubit gates.
* Fast measurement of quantum state.
