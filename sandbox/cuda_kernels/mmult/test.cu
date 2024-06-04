#include "mmult.hpp"
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_ERR 1e-4
#define THREADS_PER_BLOCK                                                      \
  1024 // Maximum number of threads per block, and per dimension is 1024

int main() {

  int m = 1 << 10;
  int k = 1 << 10; // BUG: for k greater than 1024, the kernel will not work
  int n = 1 << 10;

  float *h_a, *h_b, *h_out, *h_out_mkl;
  float *d_a, *d_b, *d_out;

  // Allocate host memory
  h_a = (float *)malloc(sizeof(float) * m * k);
  h_b = (float *)malloc(sizeof(float) * k * n);
  h_out = (float *)malloc(sizeof(float) * m * n);
  h_out_mkl = (float *)malloc(sizeof(float) * m * n);

  // Initialize host arrays
  for (int i = 0; i < m * k; i++) {
    h_a[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < k * n; i++) {
    h_b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  // Allocate device memory
  cudaMalloc((void **)&d_a, sizeof(float) * m * k);
  cudaMalloc((void **)&d_b, sizeof(float) * k * n);
  cudaMalloc((void **)&d_out, sizeof(float) * m * n);

  // Transfer data from host to device memory
  cudaMemcpy(d_a, h_a, sizeof(float) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(float) * k * n, cudaMemcpyHostToDevice);

  // Executing kernel
  custom::mmult_naive(d_out, d_a, d_b, m, k, n);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Transfer data back to host memory
  cudaMemcpy(h_out, d_out, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  custom::mmult_mkl_baseline(h_out_mkl, h_a, h_b, m, k, n);

  // Verification
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float temp = 0.0f;
      for (int l = 0; l < k; l++) {
        temp += h_a[i * k + l] * h_b[l * n + j];
      }
      assert(fabs(h_out[i * n + j] - temp) < MAX_ERR);
    }
  }
  printf("Sanity check passed\n");

  for (int i = 0; i < m * n; i++) {
    if (i < 10)
      printf("Cuda: %f | MKL: %f\n", h_out[i], h_out_mkl[i]);
    assert(fabs(h_out[i] - h_out_mkl[i]) < 0.01);
  }
  printf("MKL check passed\n");

  // Deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);

  // Deallocate host memory
  free(h_a);
  free(h_b);
  free(h_out);
  free(h_out_mkl);
}
