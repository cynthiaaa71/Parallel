%%cu
#include <cstdio>
#include <iostream>
#include <stdio.h>

#define M 800
#define N 700
#define K 600

// Kernel function to perform matrix multiplication
__global__ void matrixMulKernel(int *a, int *b, int *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        int Pvalue = 0;
        for (int i = 0; i < n; ++i) {
            Pvalue += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = Pvalue;
    }
}

int main() {
    
    clock_t start_time, end_time;
    double elapsed_time;

    // Record start time
    start_time = clock();

    int a[M][N], b[N][K], c[M][K]; 
    int *ds_a, *ds_b, *ds_c;   
    int size_a = M * N * sizeof(int); 
    int size_b = N * K * sizeof(int); 
    int size_c = M * K * sizeof(int); 

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


    // Allocating memory on device
    cudaMalloc((void**)&ds_a, size_a);
    cudaMalloc((void**)&ds_b, size_b);
    cudaMalloc((void**)&ds_c, size_c);

    // Copying input matrices from host to device
    cudaMemcpy(ds_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(ds_b, b, size_b, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(K, M);
    dim3 dimGrid(1, 1);

    // Launch kernel function for matrix multiplication
    matrixMulKernel<<<dimGrid, dimBlock>>>(ds_a, ds_b, ds_c, M, N, K);

    // Copy input matrices from device to host
    cudaMemcpy(c, ds_c, size_c, cudaMemcpyDeviceToHost);

    // Record end time
    end_time = clock(); 

    // Calculate elapsed time
    elapsed_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    // Print elapsed time
    printf("Elapsed time: %f seconds\n", elapsed_time);


    // Free device memory
    cudaFree(ds_a);
    cudaFree(ds_b);
    cudaFree(ds_c);

    return 0;
}