#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000000
#define MAX_ERR 1e-6
#define THREADS_PER_BLOCK 1024 // Maximum number of threads per block, and per dimension is 1024

__global__ void vector_add(float *out, float *a, float *b, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < n)
    out[index] = a[index] + b[index];
}

int main() {
  float *h_a, *h_b, *h_out;
  float *d_a, *d_b, *d_out;

  int blocks = 1 + ceil(N / THREADS_PER_BLOCK);

  printf("threads_per_block = %d\n", THREADS_PER_BLOCK);
  printf("blocks = %d\n", blocks);

  // Allocate host memory
  h_a = (float *)malloc(sizeof(float) * N);
  h_b = (float *)malloc(sizeof(float) * N);
  h_out = (float *)malloc(sizeof(float) * N);

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  // Allocate device memory
  cudaMalloc((void **)&d_a, sizeof(float) * N);
  cudaMalloc((void **)&d_b, sizeof(float) * N);
  cudaMalloc((void **)&d_out, sizeof(float) * N);

  // Transfer data from host to device memory
  cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Executing kernel
  vector_add<<<blocks, THREADS_PER_BLOCK>>>(d_out, d_a, d_b, N);

  // Transfer data back to host memory
  cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // Verification
  printf("out[0] = %f\n", h_out[0]);
  for (int i = 0; i < N; i++) {
    assert(fabs(h_out[i] - h_a[i] - h_b[i]) < MAX_ERR);
  }
  printf("PASSED\n");

  // Deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);

  // Deallocate host memory
  free(h_a);
  free(h_b);
  free(h_out);
}
