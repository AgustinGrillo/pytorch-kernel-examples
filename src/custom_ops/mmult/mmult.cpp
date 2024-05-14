// #include <torch/script.h>
#include <iostream>
#include <torch/torch.h>

// Custom matrix multiplication (Strided)
torch::Tensor mmult_passthrough(const torch::Tensor &a,
                                const torch::Tensor &b) {

  TORCH_CHECK(a.size(1) == b.size(0),
              "Matrix sizes do not match for multiplication");

  torch::Tensor output = torch::mm(a, b);
  return output;
}

torch::Tensor mmult_naive(const torch::Tensor &a, const torch::Tensor &b) {

  TORCH_CHECK(a.size(1) == b.size(0),
              "Matrix sizes do not match for multiplication");

  torch::Tensor output = torch::empty({a.size(0), b.size(1)});
  for (int i = 0; i < a.size(0); i++) {
    for (int j = 0; j < b.size(1); j++) {
      float sum = 0;
      for (int k = 0; k < a.size(1); k++) {
        sum += a[i][k].item<float>() * b[k][j].item<float>();
      }
      output[i][j] = sum;
    }
  }
  return output;
}

TORCH_LIBRARY(custom_ops, m) {
  m.def("mmult_passthrough(Tensor self, Tensor other) -> Tensor");
  m.def("mmult_naive(Tensor self, Tensor other) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_ops, CPU, m) {
  m.impl("mmult_passthrough", mmult_passthrough);
  m.impl("mmult_naive", mmult_naive);
}

// No Autograd support
TORCH_LIBRARY_IMPL(custom_ops, Autograd, m) {
  m.impl("mmult_passthrough",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("mmult_naive", torch::autograd::autogradNotImplementedFallback());
}
