#include <algorithm>
#include <chrono>
#include <iostream>
#include <math.h>
#include <mkl.h>
#include <numeric>
#include <omp.h>
#include <stdlib.h>
#include <thread>
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

void mmult_mkl_baseline(float *out, float *a, float *b, int m, int k, int n) {
  int lda = k;
  int ldb = n;
  int ldc = n;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, lda,
              b, ldb, 0.0f, out, ldc);
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

  mkl_set_dynamic(0);
  mkl_set_num_threads(10);
  std::cout << "omp_get_max_threads(): " << omp_get_max_threads() << std::endl;
  std::cout << "mkl_get_max_threads(): " << mkl_get_max_threads() << std::endl;

  // Matrix multiplication sizes
  int iterations = 100;
  int num_samples = 10;

  int m = 8;
  std::vector<int> ks = {
      static_cast<int>(pow(2, 3)),  static_cast<int>(pow(2, 5)),
      static_cast<int>(pow(2, 7)),  static_cast<int>(pow(2, 9)),
      static_cast<int>(pow(2, 11)), static_cast<int>(pow(2, 13)),
      static_cast<int>(pow(2, 16))};
  int n = 128;

  // Benchmark
  for (int k : ks) {
    std::cout << "\nMultiplication size: [" << m << " x " << k << "] * [" << k
              << " x " << n << "]" << std::endl;

    float *h_a, *h_b, *h_out_mkl;

    // Allocate host memory
    h_a = (float *)malloc(sizeof(float) * m * k);
    h_b = (float *)malloc(sizeof(float) * k * n);
    h_out_mkl = (float *)malloc(sizeof(float) * m * n);

    // Initialize host arrays
    for (size_t i = 0; i < m * k; i++) {
      h_a[i] = ((float)rand() / (float)RAND_MAX);
    }
    for (size_t i = 0; i < k * n; i++) {
      h_b[i] = ((float)rand() / (float)RAND_MAX);
    }

    // MKL
    std::cout << "\n--- MKL ---" << std::endl;
    std::vector<double> times_mkl = {};
    benchmark_operator(
        [&]() { mmult_mkl_baseline(h_out_mkl, h_a, h_b, m, k, n); }, times_mkl,
        iterations, num_samples);
    compute_metrics_and_print(times_mkl);

    // Deallocate host memory
    free(h_a);
    free(h_b);
    free(h_out_mkl);
  }
}
