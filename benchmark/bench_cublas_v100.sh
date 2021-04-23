nvcc blas.cu -o blas -lcublas -O3 -DN_QUBIT=24
echo N_QUBIT=24
CUDA_VISIBLE_DEVICES=0 ./blas | tee logs/cublas-v100.log

nvcc blas.cu -o blas -lcublas -O3 -DN_QUBIT=25
echo N_QUBIT=25
CUDA_VISIBLE_DEVICES=0 ./blas | tee -a logs/cublas-v100.log

nvcc blas.cu -o blas -lcublas -O3 -DN_QUBIT=26
echo N_QUBIT=26
CUDA_VISIBLE_DEVICES=0 ./blas | tee -a logs/cublas-v100.log

nvcc blas.cu -o blas -lcublas -O3 -DN_QUBIT=27
echo N_QUBIT=27
CUDA_VISIBLE_DEVICES=0 ./blas | tee -a logs/cublas-v100.log

nvcc blas.cu -o blas -lcublas -O3 -DN_QUBIT=28
echo N_QUBIT=28
CUDA_VISIBLE_DEVICES=0 ./blas | tee -a logs/cublas-v100.log
