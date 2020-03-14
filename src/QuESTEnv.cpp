#include "QuEST.h"

#include <cstdio>

QuESTEnv createQuESTEnv() {
    return QuESTEnv();
}

void destroyQuESTEnv(QuESTEnv& env) {
    printf("destroy env: just return!");
}