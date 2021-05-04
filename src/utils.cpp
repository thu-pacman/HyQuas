#include "utils.h"

#include <cstring>
#include "logger.h"

namespace MyGlobalVars {
int numGPUs;
int localGPUs;
int bit;
std::unique_ptr<cudaStream_t[]> streams;
std::unique_ptr<cudaStream_t[]> streams_comm;
std::unique_ptr<cublasHandle_t[]> blasHandles;
std::unique_ptr<cudaEvent_t[]> events;
#if USE_MPI
std::unique_ptr<ncclComm_t[]> ncclComms;
#endif

void init() {
    checkCudaErrors(cudaGetDeviceCount(&localGPUs));
    #if USE_MPI
        numGPUs = MyMPI::commSize * localGPUs;
    #else
        numGPUs = localGPUs;
    #endif
    Logger::add("Local GPU: %d", localGPUs);
    bit = get_bit(numGPUs);

    streams = std::make_unique<cudaStream_t[]>(MyGlobalVars::localGPUs);
    streams_comm = std::make_unique<cudaStream_t[]>(MyGlobalVars::localGPUs);
    blasHandles = std::make_unique<cublasHandle_t[]>(MyGlobalVars::localGPUs);
    events = std::make_unique<cudaEvent_t[]>(MyGlobalVars::localGPUs);
    checkCuttErrors(cuttInit());
    for (int i = 0; i < localGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        Logger::add("[%d] %s", i, prop.name);
        for (int j = 0; j < localGPUs; j++)
            if (i != j && (i ^ j) < 4) {
                checkCudaErrors(cudaDeviceEnablePeerAccess(j, 0));
            }
        checkCudaErrors(cudaStreamCreate(&streams[i]);)
        checkBlasErrors(cublasCreate(&blasHandles[i]));
        checkBlasErrors(cublasSetStream(blasHandles[i], streams[i]));
        checkCudaErrors(cudaStreamCreate(&streams_comm[i]));
        checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        checkCudaErrors(cudaEventCreate(&events[i]));
    }
    #if USE_MPI
        checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
        ncclUniqueId id;
        if (MyMPI::rank == 0)
            checkNCCLErrors(ncclGetUniqueId(&id));
        checkMPIErrors(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
        ncclComms = std::make_unique<ncclComm_t[]>(MyGlobalVars::localGPUs);
        checkNCCLErrors(ncclGroupStart());
        for (int i = 0; i < localGPUs; i++) {
            checkCudaErrors(cudaSetDevice(i));
            checkNCCLErrors(ncclCommInitRank(&ncclComms[i], numGPUs, id, MyMPI::rank * localGPUs + i));
        }
        checkNCCLErrors(ncclGroupEnd());
    #endif
}
};

namespace MyMPI {
int rank;
int commSize;
int commBit;
void init() {
#if USE_MPI
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
#endif
}
};


qreal zero_wrapper(qreal x) {
    const qreal eps = 1e-14;
    if (x > -eps && x < eps) {
        return 0;
    } else {
        return x;
    }
}

qComplex operator * (const qComplex& a, const qComplex& b) {
    return make_qComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

qComplex operator + (const qComplex& a, const qComplex& b) {
    return make_qComplex(a.x + b.x, a.y + b.y);
}

bool isUnitary(std::unique_ptr<qComplex[]>& mat, int n) {
    qComplex result[n * n];
    memset(result, 0, sizeof(result));
    for (int k = 0; k < n; k++)
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                qComplex v1 = mat[k * n + i];
                v1.y = - v1.y;
                result[i * n + j] = result[i * n + j] + v1 * mat[k * n + j];
            }
    bool wa = 0;
    qreal eps = 1e-8;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        qComplex val = result[i * n + i];
        if (fabs(val.x - 1) > eps || fabs(val.y) > eps) {
            wa = 1;
        }
        for (int j = 0; j < n; j++) {
            if (i == j)
                continue;
            qComplex val = result[i * n + j];
            if (fabs(val.x) > eps || fabs(val.y) > eps)
                wa = 1;
        }
    }
    if (wa) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("(%.2f %.2f) ", result[i * n + j].x, result[i * n + j].y);
            printf("\n");
        }
        exit(1);
    }
    return 1;
}

qComplex make_qComplex(qreal x) {
    return make_qComplex(x, 0.0);
}

bool operator < (const qComplex& a, const qComplex& b) {
        return a.x == b.x ? a.y < b.y : a.x < b.x;
}

int get_bit(int n) {
    int x = n;
    int bit = -1;
    while (x) {
        bit ++;
        x >>= 1;
    }
    if (n == 0 || (1 << bit) != n) {
        printf("Must be pow of two: %d\n", n);
        exit(1);
    }
    return bit;
}