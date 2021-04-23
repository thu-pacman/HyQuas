nvcc blas.cu -o blas -lcublas -O3 -DN_QUBIT=26
echo N_QUBIT=26
CUDA_VISIBLE_DEVICES=0 ./blas | tee logs/cublas-a100.log

nvcc blas.cu -o blas -lcublas -O3 -DN_QUBIT=27
echo N_QUBIT=27
CUDA_VISIBLE_DEVICES=0 ./blas | tee logs/cublas-a100.log

nvcc blas.cu -o blas -lcublas -O3 -DN_QUBIT=28
echo N_QUBIT=28
CUDA_VISIBLE_DEVICES=0 ./blas | tee logs/cublas-a100.log

nvcc blas.cu -o blas -lcublas -O3 -DN_QUBIT=29
echo N_QUBIT=29
CUDA_VISIBLE_DEVICES=0 ./blas | tee logs/cublas-a100.log

nvcc blas.cu -o blas -lcublas -O3 -DN_QUBIT=30
echo N_QUBIT=30
CUDA_VISIBLE_DEVICES=0 ./blas | tee logs/cublas-a100.log
