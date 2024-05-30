#include <torch/torch.h>

torch::Tensor mmult_mkl(const torch::Tensor &a, const torch::Tensor &b);

torch::Tensor mmult_naive(const torch::Tensor &a, const torch::Tensor &b);

torch::Tensor mmult_naive_multithreaded(const torch::Tensor &a,
                                        const torch::Tensor &b);
