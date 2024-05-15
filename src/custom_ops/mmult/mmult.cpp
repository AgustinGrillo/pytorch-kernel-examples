#include <iostream>
#include <mkl.h>
#include <torch/torch.h>

/**
 *
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

/* Python bindings */
TORCH_LIBRARY(custom_ops, m) {
  m.def("mmult_passthrough(Tensor self, Tensor other) -> Tensor");
  m.def("mmult_naive(Tensor self, Tensor other) -> Tensor");
  m.def("mmult_mkl(Tensor self, Tensor other) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_ops, CPU, m) {
  m.impl("mmult_passthrough", mmult_passthrough);
  m.impl("mmult_naive", mmult_naive);
  m.impl("mmult_mkl", mmult_mkl);
}

// No Autograd support
TORCH_LIBRARY_IMPL(custom_ops, Autograd, m) {
  m.impl("mmult_passthrough",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("mmult_naive", torch::autograd::autogradNotImplementedFallback());
  m.impl("mmult_mkl", torch::autograd::autogradNotImplementedFallback());
}
