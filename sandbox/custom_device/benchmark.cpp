#include "custom_backend.hpp"
#include <iostream>
#include <math.h>

void benchmark() {
  // Benchmarking code

  int iterations = 100;
  // Matrix sizes
  int m = 8;
  int k = pow(2, 15);
  int n = 128;

  torch::Tensor a = torch::randn({m, k});
  torch::Tensor b = torch::randn({k, n});

  // Warm up
  for (int i = 0; i < iterations; i++) {
    vgpu::mm(a, b);
  }

  // Benchmark (compute times for each iteration)
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    vgpu::mm(a, b);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  // duration in microseconds
  std::chrono::duration<double, std::micro> duration_us = duration;
  double mean_us = duration_us.count() / iterations;
  std::cout << "Mean: " << mean_us << " us" << std::endl;
}

int main() {
  benchmark();
  return 0;
}
