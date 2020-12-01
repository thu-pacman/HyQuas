#include "utils.h"

namespace MyMPI {
int rank;
int commSize;
int commBit;
void init() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    commBit = -1;
    int x = commSize;
    while (x) {
        commBit ++;
        x >>= 1;
    }
    if ((1 << commBit) != commSize) {
        printf("Invalid Comm Size! %d %d\n", commBit, commSize);
        exit(1);
    }
}
};
