#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>

// maximum size of matrix
#define M 1000
#define N 1000
#define K 1000

// maximum number of threads
#define MAX_THREAD 4

int matA[M][N];
int matB[N][K];
int matC[M][K];
int step_i = 0;
pthread_mutex_t lock;
clock_t start_time, end_time;

void* multi(void* arg) {
    int start, end;

    // Lock to ensure that step_i is accessed by one thread at a time
    pthread_mutex_lock(&lock);
    start = step_i;
    step_i += M / MAX_THREAD;
    pthread_mutex_unlock(&lock);
    end = step_i;
    // For the last thread, handle any remaining rows
    if (step_i+(M / MAX_THREAD)>= M) {
        end = M;
    }
    
    // i denotes row number of resultant matC
    for (int i = start; i < end; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < N; k++) {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }

    return NULL;
}



// Driver Code
int main() {
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
    
     if (pthread_mutex_init(&lock, NULL) != 0) {
        printf("\n Mutex initialization failed.\n");
        return 1;
    }
    // declaring four threads
    pthread_t threads[MAX_THREAD];
    start_time = clock();

    // Creating M threads, each evaluating its own part
    for (int i = 0; i < MAX_THREAD; i++) {
        pthread_create(&threads[i], NULL, multi, NULL);
    }

    // joining and waiting for all threads to complete
    for (int i = 0; i < MAX_THREAD; i++)
        pthread_join(threads[i], NULL);
        
    end_time = clock();
    double elapsed_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    // Displaying the result matrix
    /*printf("\nMultiplication of A and B\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++)
            printf("%d ", matC[i][j]);
        printf("\n");
    }*/
       printf("Elapsed time: %.7f seconds\n", elapsed_time);

    pthread_mutex_destroy(&lock);
    return 0;
}