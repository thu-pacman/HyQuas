#include "QuEST.h"
#include "stdio.h"
#include "mytimer.hpp"


int main (int narg, char *argv[]) {

    QuESTEnv Env = createQuESTEnv();
    double t1 = get_wall_time();

    FILE *fp=fopen("probs.dat", "w");
    if(fp==NULL){
        printf("    open probs.dat failed, Bye!");
        return 0;
    }

    FILE *fvec=fopen("stateVector.dat", "w");
    if(fp==NULL){
        printf("    open stateVector.dat failed, Bye!");
        return 0;
    }

    Qureg q = createQureg(30, Env);

    float q_measure[30];
    hadamard(q, 0);
    hadamard(q, 1);
    hadamard(q, 2);
    hadamard(q, 3);
    hadamard(q, 4);
    hadamard(q, 5);
    hadamard(q, 6);
    hadamard(q, 7);
    hadamard(q, 8);
    hadamard(q, 9);
    hadamard(q, 10);
    hadamard(q, 11);
    hadamard(q, 12);
    hadamard(q, 13);
    hadamard(q, 14);
    hadamard(q, 15);
    hadamard(q, 16);
    hadamard(q, 17);
    hadamard(q, 18);
    hadamard(q, 19);
    hadamard(q, 20);
    hadamard(q, 21);
    hadamard(q, 22);
    hadamard(q, 23);
    hadamard(q, 24);
    hadamard(q, 25);
    hadamard(q, 26);
    hadamard(q, 27);
    hadamard(q, 28);
    hadamard(q, 29);
    q.compile();
    double t3 = get_wall_time();
    q.run();
    double t4 = get_wall_time();
    printf("run %12.6f\n", t4 - t3);
    printf("\n");
    for(long long int i=0; i<29; ++i){
        q_measure[i] = calcProbOfOutcome(q,  i, 1);
        //printf("  probability for q[%2lld]==1 : %lf    \n", i, q_measure[i]);
        fprintf(fp, "Probability for q[%2lld]==1 : %lf    \n", i, q_measure[i]);
    }
    fprintf(fp, "\n");
    printf("\n");


    for(int i=0; i<10; ++i){
        Complex amp = getAmp(q, i);
        //printf("Amplitude of %dth state vector: %f\n", i, prob);
	fprintf(fvec, "Amplitude of %dth state vector: %12.6f,%12.6f\n", i, amp.real, amp.imag);
    }

    double t2 = get_wall_time();
    printf("Complete the simulation takes time %12.6f seconds.", t2 - t1);
    printf("\n");
    destroyQureg(q, Env);
    destroyQuESTEnv(Env);

    return 0;
}
