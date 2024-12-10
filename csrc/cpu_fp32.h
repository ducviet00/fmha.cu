#include <torch/torch.h>

void cpu_fmha_fp32(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O);
