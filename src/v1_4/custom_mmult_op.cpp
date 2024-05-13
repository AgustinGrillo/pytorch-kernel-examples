// #include <torch/script.h>
#include <iostream>
// #include <torch/extension.h>
#include <torch/torch.h>

#define DEBUG

torch::Tensor custom_mmult(torch::Tensor a, torch::Tensor b) {

  TORCH_CHECK(a.size(1) == b.size(0),
              "Matrix sizes do not match for multiplication");

#ifdef DEBUG
  std::cout << "Operation running inside custom kernel " << std::endl;
#endif

  torch::Tensor output = torch::zeros({a.size(0), b.size(1)});
  torch::mm_out(output, a, b);
  return output.clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("custom_mmult", &custom_mmult, "Custom matrix multiplication");
}
