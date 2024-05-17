#include <iostream>
#include <mkl.h>
#include <thread>
#include <torch/torch.h>

// Global variables (for demonstration purposes)
int64_t num_threads = 1;

/**
 * Matrix multiplication using MKL
 */
torch::Tensor mmult_mkl(const torch::Tensor &a, const torch::Tensor &b) {

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
  torch::Tensor output = torch::zeros({m, n});

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
 * Matrix multiplication using torch::mm
 */
torch::Tensor mmult_passthrough(const torch::Tensor &a,
                                const torch::Tensor &b) {

  TORCH_CHECK(a.size(1) == b.size(0),
              "Matrix sizes do not match for multiplication");

  torch::Tensor output = torch::mm(a, b);
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
 * Set the number of threads for multithreaded operations
 */
void set_num_threads(int64_t threads) {
  num_threads = threads;
#ifdef DEBUG
  std::cout << "Setting " << num_threads << " threads" << std::endl;
#endif
}

void set_thread_affinity_for_core(int i) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(i, &cpuset);
  auto s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (s != 0) {
    std::cerr << "Error setting thread affinity for thread " << i << std::endl;
  }
}

/**
 * Multithreaded (Naive) Matrix multiplication (column parallelism)
 */
torch::Tensor mmult_naive_multithreaded(const torch::Tensor &a,
                                        const torch::Tensor &b) {

#ifdef DEBUG
  std::cout << "Running with " << num_threads << " threads" << std::endl;
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
  for (int i = 0; i < num_threads; i++) {
    std::thread t([i, n, ac, bc, output]() {
      // // Set thread affinity
      // set_thread_affinity_for_core(i);
      // Compute mmult for the slice
      int start = i * n / num_threads;
      int end = (i + 1) * n / num_threads;
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
TORCH_LIBRARY(custom_ops, m) {
  m.def("mmult_passthrough(Tensor self, Tensor other) -> Tensor");
  m.def("mmult_naive(Tensor self, Tensor other) -> Tensor");
  m.def("mmult_naive_multithreaded(Tensor self, Tensor other) -> Tensor");
  m.def("mmult_mkl(Tensor self, Tensor other) -> Tensor");
  m.def("set_num_threads", set_num_threads);
}
// m.def(TORCH_SELECTIVE_SCHEMA("custom_ops::op(...) ->  (...)"));

TORCH_LIBRARY_IMPL(custom_ops, CPU, m) {
  m.impl("mmult_passthrough", mmult_passthrough);
  m.impl("mmult_naive", mmult_naive);
  m.impl("mmult_naive_multithreaded", mmult_naive_multithreaded);
  m.impl("mmult_mkl", mmult_mkl);
}
// m.impl(TORCH_SELECTIVE_NAME("custom_ops::op"), TORCH_FN(op));

// No Autograd support
TORCH_LIBRARY_IMPL(custom_ops, Autograd, m) {
  m.impl("mmult_passthrough",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("mmult_naive", torch::autograd::autogradNotImplementedFallback());
  m.impl("mmult_naive_multithreaded",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("mmult_mkl", torch::autograd::autogradNotImplementedFallback());
}

// // Fallback
// TORCH_LIBRARY_IMPL(_, /*DispatchKey=*/vGPU, m) {
//   m.fallback(torch::CppFunction::makeFallthrough());
// }
