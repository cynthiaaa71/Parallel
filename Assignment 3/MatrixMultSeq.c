

// C program to multiply two matrices
#include <stdio.h>
#include <stdlib.h>
 
#define M 900
#define N 700
#define K 600

 
void multiplyMatrix(int m1[M][N], int m2[N][K])
{
    int result[M][K];
 
    printf("Resultant Matrix is:\n");
 
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            result[i][j] = 0;
 
            for (int k = 0; k < N; k++) {
                result[i][j] += m1[i][k] * m2[k][j];
            }

        }
    }
}
 
// Driver code
int main()
{
    int a[M][N];
    int b[N][K];
    clock_t start_time, end_time;
    double elapsed_time;

    // Record start time
    start_time = clock();
    // Initialize input matrices a
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i][j] = i + j;
        }
    }
    // Initialize input matrices b
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            b[i][j] = i - j;
        }
    }
        
    multiplyMatrix(a, b);
    
    // Record end time
    end_time = clock(); 
     // Calculate elapsed time
    elapsed_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    // Print elapsed time
    printf("Elapsed time: %f seconds\n", elapsed_time);
 
    return 0;
}