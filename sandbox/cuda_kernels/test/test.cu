#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 3

__global__ void plus_one(float *a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    a[idx] += 1;
  }
}

int main() {
  float *h_a;
  float *d_a;

  // Allocate host memory
  h_a = (float *)malloc(sizeof(float) * N);

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    h_a[i] = 0.0f;
    printf("h_a[%d] = %f\n", i, h_a[i]);
  }

  // Allocate device memory
  cudaMalloc((void **)&d_a, sizeof(float) * N);

  // Transfer data from host to device memory
  cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Executing kernel
  plus_one<<<1, N>>>(d_a, N);

  // Transfer data back to host memory
  cudaMemcpy(h_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // Verification
  for (int i = 0; i < N; i++) {
    printf("h_a[%d] = %f\n", i, h_a[i]);
  }

  // Deallocate device memory
  cudaFree(d_a);

  // Deallocate host memory
  free(h_a);
}
