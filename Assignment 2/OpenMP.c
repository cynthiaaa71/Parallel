#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#define M 1000
#define N 1000
#define K 1000

int matA[M][N];
int matB[N][K];
int matC[M][K];
clock_t start_time, end_time;

int main() {
    int i, j, k;

    // Setting the number of OpenMP threads to the number of available processors
    omp_set_num_threads(omp_get_num_procs());

    // Generating random values in matA and matB
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matA[i][j] = rand() % 10;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            matB[i][j] = rand() % 10;
        }
    }

    // Displaying matA
    /*printf("\nMatrix A\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", matA[i][j]);
        printf("\n");
    }

    // Displaying matB
    printf("\nMatrix B\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++)
            printf("%d ", matB[i][j]);
        printf("\n");
    }*/

    // Recording the start time
    start_time = clock();

    // OpenMP parallelized matrix multiplication
    #pragma omp parallel for private(i, j, k) shared(matA, matB, matC)
    for (i = 0; i < M; ++i) {
        for (j = 0; j < K; ++j) {
            for (k = 0; k < N; ++k) {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
     // Displaying matC
    /*printf("\nMatrix C\n");
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++)
          printf("%d ", matC[i][j]);
      printf("\n");
    }
*/

    // Recording the end time
    end_time = clock();
    double elapsed_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;


    // Calculating and printing the elapsed time
    printf("elapsed time = %.7f seconds.\n", elapsed_time);

    return 0;
}
