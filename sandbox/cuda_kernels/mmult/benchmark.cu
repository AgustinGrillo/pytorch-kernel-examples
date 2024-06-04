#include "mmult.hpp"
#include <algorithm>
#include <chrono>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <numeric>
#include <stdlib.h>
#include <vector>

/**
 * Compute the median of a vector
 */
double median(const std::vector<double> &v) {
  std::vector<double> temp(v);
  size_t n = temp.size() / 2;
  nth_element(temp.begin(), temp.begin() + n, temp.end());
  return temp[n];
}

void compute_metrics_and_print(const std::vector<double> &times) {
  double min_time = *std::min_element(times.begin(), times.end());
  double median_time = median(times);
  double mean_time =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();

  std::cout << "Min: " << min_time << " us" << std::endl;
  std::cout << "Median: " << median_time << " us" << std::endl;
  std::cout << "Mean: " << mean_time << " us" << std::endl;
}

void benchmark_operator(const std::function<void()> &callable_op,
                        std::vector<double> &times, int iterations,
                        int num_samples) {
  for (int ns = 0; ns < num_samples; ns++) {
    // Warm up
    for (int it = 0; it < iterations; it++) {
      callable_op();
    }
    // Benchmark (compute times for each iteration)
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; it++) {
      callable_op();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    // duration in microseconds
    std::chrono::duration<double, std::micro> duration_us = duration;
    double mean_us = duration_us.count() / iterations;
    times.push_back(mean_us);
  }
}

/**
 * Benchmark matrix multiplication
 */
int main() {
  srand(0);

  // Matrix multiplication sizes
  int iterations = 1000;
  int num_samples = 100;
  int m = 8;
  std::vector<int> ks = {
      static_cast<int>(pow(2, 3)), static_cast<int>(pow(2, 5)),
      static_cast<int>(pow(2, 7)), static_cast<int>(pow(2, 10))};
  int n = 128;

  // Device
  cudaDeviceProp deviceProp;
  int deviceIdx = 0;
  cudaSetDevice(deviceIdx);
  cudaGetDeviceProperties(&deviceProp, deviceIdx);
  printf("Device %d: %s\n", deviceIdx, deviceProp.name);
  int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
  printf("enable_tf32: %d\n", enable_tf32);

  // Create cuBLAS handle
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cublasMath_t cublas_math_mode =
      enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasSetMathMode(cublas_handle, cublas_math_mode);

  // Benchmark
  for (int k : ks) {
    std::cout << "\nMultiplication size: [" << m << " x " << k << "] * [" << k
              << " x " << n << "]" << std::endl;

    float *h_a, *h_b, *h_out, *h_out_mkl;
    float *d_a, *d_b, *d_out;

    // Allocate host memory
    h_a = (float *)malloc(sizeof(float) * m * k);
    h_b = (float *)malloc(sizeof(float) * k * n);
    h_out = (float *)malloc(sizeof(float) * m * n);
    h_out_mkl = (float *)malloc(sizeof(float) * m * n);

    // Initialize host arrays
    for (size_t i = 0; i < m * k; i++) {
      h_a[i] = ((float)rand() / (float)RAND_MAX);
    }
    for (size_t i = 0; i < k * n; i++) {
      h_b[i] = ((float)rand() / (float)RAND_MAX);
    }
    // Allocate device memory
    cudaMalloc(&d_a, sizeof(float) * m * k);
    cudaMalloc(&d_b, sizeof(float) * k * n);
    cudaMalloc(&d_out, sizeof(float) * m * n);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * k * n, cudaMemcpyHostToDevice);

    // TODO: Check output correctness

    // MKL
    std::cout << "\n--- MKL ---" << std::endl;
    std::vector<double> times_mkl = {};
    benchmark_operator(
        [&]() { custom::mmult_mkl_baseline(h_out_mkl, h_a, h_b, m, k, n); },
        times_mkl, iterations, num_samples);
    compute_metrics_and_print(times_mkl);

    // CUDA Naive
    std::cout << "\n--- CUDA Naive ---" << std::endl;
    std::vector<double> times_cuda_naive = {};
    benchmark_operator(
        [&]() {
          custom::mmult_naive(d_out, d_a, d_b, m, k, n);
          cudaDeviceSynchronize();
        },
        times_cuda_naive, iterations, num_samples);
    compute_metrics_and_print(times_cuda_naive);

    // CUDA cuBLAS
    std::cout << "\n--- CUDA cuBLAS ---" << std::endl;
    std::vector<double> times_cuda_cublas = {};
    benchmark_operator(
        [&]() {
          custom::mmult_cublas(cublas_handle, d_out, d_a, d_b, m, k, n);
          cudaDeviceSynchronize();
        },
        times_cuda_cublas, iterations, num_samples);
    compute_metrics_and_print(times_cuda_cublas);

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(h_a);
    free(h_b);
    free(h_out);
    free(h_out_mkl);
    cublasDestroy(cublas_handle);
  }
}
