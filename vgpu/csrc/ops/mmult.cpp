#include "mmult.hpp"
#include "../utils/multithreading.hpp"
#include <iostream>
#include <mkl.h>

/**
 * Matrix multiplication using MKL
 */
torch::Tensor mmult_mkl(const torch::Tensor &a, const torch::Tensor &b) {

  TORCH_CHECK(a.size(1) == b.size(0),
              "Matrix sizes do not match for multiplication");

  float alpha = 1.0f;
  float beta = 0.0f;

  int m = a.size(0);
  int k = a.size(1);
  int n = b.size(1);

  int lda = k;
  int ldb = n;
  int ldc = n;

  // mkl_set_num_threads(mkl_get_max_threads());

  // C = (double *)mkl_malloc( m*n*sizeof( double ), 64 ); // mkl_free(C)
  torch::Tensor output = torch::zeros({m, n}, a.options());

  torch::Tensor ac = a.contiguous();
  torch::Tensor bc = b.contiguous();
  const float *ac_ptr = ac.data_ptr<float>();
  const float *bc_ptr = bc.data_ptr<float>();
  float *output_ptr = output.data_ptr<float>();

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, ac_ptr,
              lda, bc_ptr, ldb, beta, output_ptr, ldc);

  return output;
}

/**
 * Naive matrix multiplication implementation
 */
torch::Tensor mmult_naive(const torch::Tensor &a, const torch::Tensor &b) {

  TORCH_CHECK(a.size(1) == b.size(0),
              "Matrix sizes do not match for multiplication");

  torch::TensorAccessor<float, 2> a_acc = a.accessor<float, 2>();
  torch::TensorAccessor<float, 2> b_acc = b.accessor<float, 2>();
  torch::Tensor output = torch::empty({a.size(0), b.size(1)});

  for (int i = 0; i < a.size(0); i++) {
    for (int j = 0; j < b.size(1); j++) {
      float sum = 0;
      for (int k = 0; k < a.size(1); k++) {
        sum += a_acc[i][k] * b_acc[k][j];
      }
      output[i][j] = sum;
    }
  }
  return output;
}

/**
 * Multithreaded (Naive) Matrix multiplication (column parallelism)
 */
torch::Tensor mmult_naive_multithreaded(const torch::Tensor &a,
                                        const torch::Tensor &b) {

#ifdef DEBUG
  std::cout << "Running with " << vgpu::get_num_threads() << " threads"
            << std::endl;
#endif

  int m = a.size(0);
  int k = a.size(1);
  int n = b.size(1);

  torch::Tensor ac = a.contiguous();
  torch::Tensor bc = b.contiguous();

  // Create output tensor
  torch::Tensor output = torch::zeros({m, n});

  // Create threads
  std::vector<std::thread> threads;
  for (int i = 0; i < vgpu::get_num_threads(); i++) {
    std::thread t([i, n, ac, bc, output]() {
      // // Set thread affinity
      // set_thread_affinity_for_core(i);
      // Compute mmult for the slice
      int start = i * n / vgpu::get_num_threads();
      int end = (i + 1) * n / vgpu::get_num_threads();
      // slice the matrix b -> bc_i[:, start:end]
      torch::Tensor bc_i = bc.slice(1, start, end);
      torch::Tensor partial_output = mmult_naive(ac, bc_i);
      // copy the partial output to the final output
      output.index({torch::indexing::Slice(),
                    torch::indexing::Slice(start, end)}) = partial_output;
    });
    threads.push_back(std::move(t));
  }

  // Wait for all threads to finish
  for (auto &t : threads) {
    t.join();
  }

  return output;
}

/* Python bindings */
TORCH_LIBRARY_FRAGMENT(vgpu, m) {
  m.def("mmult_mkl(Tensor self, Tensor other) -> Tensor");
  m.def("mmult_naive(Tensor self, Tensor other) -> Tensor");
  m.def("mmult_naive_multithreaded(Tensor self, Tensor other) -> Tensor");
}

TORCH_LIBRARY_IMPL(vgpu, CPU, m) {
  m.impl("mmult_mkl", mmult_mkl);
  m.impl("mmult_naive", mmult_naive);
  m.impl("mmult_naive_multithreaded", mmult_naive_multithreaded);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) { m.impl("mm", mmult_mkl); }

// No Autograd support
TORCH_LIBRARY_IMPL(vgpu, Autograd, m) {
  m.impl("mmult_mkl", torch::autograd::autogradNotImplementedFallback());
  m.impl("mmult_naive", torch::autograd::autogradNotImplementedFallback());
  m.impl("mmult_naive_multithreaded",
         torch::autograd::autogradNotImplementedFallback());
}
