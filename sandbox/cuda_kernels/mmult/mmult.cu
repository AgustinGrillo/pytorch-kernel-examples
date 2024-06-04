#include "mmult.hpp"
#include <assert.h>
// #include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef MKL
#include <mkl.h>
#endif // MKL

/*
 * ----------------------------------------------------------------------------
 * Naive mmult cuda kernel
 * ----------------------------------------------------------------------------
 */
// Kernel
__global__ void mmult_naive_kernel(float *out, float *a, float *b, int m, int k,
                                   int n) {
  // Single thread per element of the output matrix
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  // a = m x k
  // b = k x n
  // out = m x n

  if (row < m && col < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
      sum += a[row * k + i] * b[i * n + col];
    }
    out[row * n + col] = sum;
  }
}

// Kernel launcher
void custom::mmult_naive(float *out, float *a, float *b, int m, int k, int n) {
  // TODO: Add block size as an argument

  dim3 block_size(8, 8);

  dim3 grid_size((m + block_size.x - 1) / block_size.x,
                 (n + block_size.y - 1) / block_size.y);

  mmult_naive_kernel<<<grid_size, block_size>>>(out, a, b, m, k, n);

  // Check error
  if (cudaGetLastError() != cudaSuccess) {
    printf("[CUDA error]: \n%s\n", cudaGetErrorString(cudaGetLastError()));
    exit(1);
  }
}

/*
 * ----------------------------------------------------------------------------
 * Cublas mmult cuda kernel
 * ----------------------------------------------------------------------------
 */
// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
    exit(EXIT_FAILURE);
  }
}
#define cublasCheck(status)                                                    \
  { cublasCheck((status), __FILE__, __LINE__); }

// Kernel
void custom::mmult_cublas(cublasHandle_t cublas_handle, float *out, float *a,
                          float *b, int m, int k, int n) {
  // API: https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm
  // cublasStatus_t cublasSgemm(cublasHandle_t handle,
  //                        cublasOperation_t transa, cublasOperation_t transb,
  //                        int m, int n, int k,
  //                        const float           *alpha,
  //                        const float           *A, int lda,
  //                        const float           *B, int ldb,
  //                        const float           *beta,
  //                        float           *C, int ldc)
  // cuBLAS does C = alpha * A * B + beta * C
  // where A is mxk, B is kxn, C is mxn
  // NOTE: cublas uses column-major storage => it sees our matrices (row-major)
  // as transposed
  // => if out = A @ B -> out.T = B.T @ A.T
  // Because cuBLAS is column-major, we actually want to get it to calculate
  // out.T

  const float alpha = 1.0f;
  const float beta = 0.0f;
  int lda = k;
  int ldb = n;
  int ldc = n;
  cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                          &alpha, b, ldb, a, lda, &beta, out, ldc));
}

/*
 * ----------------------------------------------------------------------------
 * CPU (MKL) mmult baseline
 * ----------------------------------------------------------------------------
 */
void custom::mmult_mkl_baseline(float *out, float *a, float *b, int m, int k,
                                int n) {
#ifdef MKL
  int lda = k;
  int ldb = n;
  int ldc = n;
  // mkl_set_num_threads(mkl_get_max_threads());
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, lda,
              b, ldb, 0.0f, out, ldc);
#endif // MKL
}
