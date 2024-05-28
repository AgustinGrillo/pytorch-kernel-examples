#include <mkl.h>
#include <torch/torch.h>

namespace vgpu {
torch::Tensor mm(const torch::Tensor &a, const torch::Tensor &b);
}
