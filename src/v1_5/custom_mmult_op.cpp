// #include <torch/script.h>
#include <iostream>
#include <torch/torch.h>

#define DEBUG

// Custom matrix multiplication (Strided)
torch::Tensor custom_mmult_strided(const torch::Tensor& a, const torch::Tensor& b) {

  TORCH_CHECK(a.size(1) == b.size(0),
              "Matrix sizes do not match for multiplication");

#ifdef DEBUG
  std::cout << "Operation running inside custom kernel for dense tensor"
            << std::endl;
#endif

  torch::Tensor output =
      torch::zeros({a.size(0), b.size(1)});
  torch::mm_out(output, a, b);
  return output.clone();
}

// Custom matrix multiplication (Sparse)
torch::Tensor custom_mmult_sparse(const torch::Tensor& a, const torch::Tensor& b) {

  TORCH_CHECK(a.size(1) == b.size(0),
              "Matrix sizes do not match for multiplication");

#ifdef DEBUG
  std::cout << "Operation running inside custom kernel for sparse tensor"
            << std::endl;
#endif

  torch::Tensor output =
      torch::zeros({a.size(0), b.size(1)});
  torch::mm_out(output, a, b);
  return output.clone();
}

TORCH_LIBRARY(custom_ops, m) {
  m.def("custom_mmult(Tensor self, Tensor other) -> Tensor");

  // Registers the op to be run on all cases ("catch-all" kernel)
  // m.def("custom_mmult", custom_mmult)
}

TORCH_LIBRARY_IMPL(custom_ops, CPU, m) { m.impl("custom_mmult", custom_mmult_strided); }
TORCH_LIBRARY_IMPL(custom_ops, SparseCPU, m) { m.impl("custom_mmult", custom_mmult_sparse); }

// No Autograd support
TORCH_LIBRARY_IMPL(custom_ops, Autograd, m) {
  m.impl("custom_mmult", torch::autograd::autogradNotImplementedFallback());
}
