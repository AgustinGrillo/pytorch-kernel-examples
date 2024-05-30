#include "ops/mmult.hpp"
#include <iostream>
#include <math.h>

/**
 * Compute the median of a vector
 */
double median(const std::vector<double> &v) {
  std::vector<double> temp(v);
  size_t n = temp.size() / 2;
  nth_element(temp.begin(), temp.begin() + n, temp.end());
  return temp[n];
}

/**
 * Benchmark matrix multiplication
 */
void benchmark() {

  int iterations = 1000;
  int num_samples = 100;
  int m = 8;
  std::vector<int> ks = {
      static_cast<int>(pow(2, 3)), static_cast<int>(pow(2, 7)),
      static_cast<int>(pow(2, 11)), static_cast<int>(pow(2, 15))};
  int n = 128;

  for (int k : ks) {
    std::cout << "\nMultiplication size: [" << m << " x " << k << "] * [" << k
              << " x " << n << "]" << std::endl;
    torch::Tensor a = torch::randn({m, k});
    torch::Tensor b = torch::randn({k, n});

    std::vector<double> times = {};

    for (int ns = 0; ns < num_samples; ns++) {
      // Warm up
      for (int it = 0; it < iterations; it++) {
        mmult_mkl(a, b);
      }
      // Benchmark (compute times for each iteration)
      auto start = std::chrono::high_resolution_clock::now();
      for (int it = 0; it < iterations; it++) {
        mmult_mkl(a, b);
      }
      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> duration = end - start;
      // duration in microseconds
      std::chrono::duration<double, std::micro> duration_us = duration;
      double mean_us = duration_us.count() / iterations;
      times.push_back(mean_us);
    }

    double min_time = *std::min_element(times.begin(), times.end());
    double median_time = median(times);
    double mean_time =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    std::cout << "Min: " << min_time << " us" << std::endl;
    std::cout << "Median: " << median_time << " us" << std::endl;
    std::cout << "Mean: " << mean_time << " us" << std::endl;
  }
}

int main() {
  benchmark();
  return 0;
}
