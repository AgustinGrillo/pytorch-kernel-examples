// #include <torch/script.h>
#include <iostream>
#include <torch/torch.h>

#define DEBUG

torch::Tensor custom_mmult(torch::Tensor a, torch::Tensor b) {

  TORCH_CHECK(a.size(1) == b.size(0),
              "Matrix sizes do not match for multiplication");

  // "switch" based on tensor layout
  if (a.layout() == torch::kStrided) {

#ifdef DEBUG
    std::cout << "Operation running inside custom kernel for dense matrix"
              << std::endl;
#endif
    torch::Tensor output =
        torch::zeros({a.size(0), b.size(1)}, torch::kFloat32);
    torch::mm_out(output, a, b);
    return output.clone();

  } else if (a.layout() == torch::kSparse) {

#ifdef DEBUG
    std::cout << "Operation running inside custom kernel for sparse matrix"
              << std::endl;
#endif
    torch::Tensor output = torch::zeros({a.size(0), b.size(1)});
    torch::mm_out(output, a, b);
    return output.clone();

  } else {
    TORCH_CHECK(false, "Unsupported layout for custom kernel");
  }
}

TORCH_LIBRARY(custom_ops, m) { m.def("custom_mmult", custom_mmult); }
