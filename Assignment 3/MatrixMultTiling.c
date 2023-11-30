#include <stdio.h>

#define M 1000
#define N 700
#define K 700

#define TILE_WIDTH 60

// Kernel function to perform matrix multiplication using tiling
__global__ void matrixMulKernel(int *a, int *b, int *c, int m, int n, int k) {
    __shared__ int tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ int tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int Pvalue = 0;

    for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; ++t) {
        if (row < m && (t * TILE_WIDTH + threadIdx.x) < n)
            tileA[threadIdx.y][threadIdx.x] = a[row * n + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        if ((t * TILE_WIDTH + threadIdx.y) < n && col < k)
            tileB[threadIdx.y][threadIdx.x] = b[(t * TILE_WIDTH + threadIdx.y) * k + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads(); // wait for all threads to start together

        for (int i = 0; i < TILE_WIDTH; ++i)
            Pvalue += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads(); // wait for all threads to finish
    }

    if (row < m && col < k)
        c[row * k + col] = Pvalue;
}

int main() {
    
    clock_t start_time, end_time;
    double elapsed_time;


    // Record start time
    start_time = clock();


    int a[M][N], b[N][K], c[M][K]; // Input matrices and result matrix
    int *ds_a, *ds_b, *ds_c;    // Device copies of matrices
    int size_a = M * N * sizeof(int); // Size of matrix a in bytes
    int size_b = N * K * sizeof(int); // Size of matrix b in bytes
    int size_c = M * K * sizeof(int); // Size of matrix c in bytes

    // Initialize input matrix a
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i][j] = i + j;
        }
    }
    // Initialize input matrix b
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            b[i][j] = i - j;
        }
    }

    // Allocate memory on the device
    cudaMalloc((void**)&ds_a, size_a);
    cudaMalloc((void**)&ds_b, size_b);
    cudaMalloc((void**)&ds_c, size_c);

    // Copy input matrices from host to device
    cudaMemcpy(ds_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(ds_b, b, size_b, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((K - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1);

    // Launch kernel function for matrix multiplication
    matrixMul<<<dimGrid, dimBlock>>>(ds_a, ds_b, ds_c, M, N, K);

    // Copy result matrix from device to host
    cudaMemcpy(c, ds_c, size_c, cudaMemcpyDeviceToHost);

    // Record end time
    end_time = clock();

 
    // Calculate elapsed time
    elapsed_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

 
    // Print elapsed time
    printf("Elapsed time (with Tiling): %f seconds\n", elapsed_time);

    // Free device memory
    cudaFree(ds_a);
    cudaFree(ds_b);
    cudaFree(ds_c);

    return 0;
    }