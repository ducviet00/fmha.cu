import torch
from torch.nn import functional as F

import fmha_cu


def test_cpu_correctness():
    b, h, s, d = 2, 12, 229, 32

    Q = torch.randn([b, h, s, d], dtype=torch.float32, device="cpu")
    K = torch.randn([b, h, s, d], dtype=torch.float32, device="cpu")
    V = torch.randn([b, h, s, d], dtype=torch.float32, device="cpu")

    O_cpu = fmha_cu.ops.fmha(Q, K, V)
    O_ref = F.scaled_dot_product_attention(Q, K, V)

    torch.testing.assert_close(O_cpu, O_ref, atol=5e-4, rtol=0)


def test_multipass_cuda_fp16_correctness():
    b, h, s, d = 2, 24, 3229, 64

    Q = torch.randn([b, h, s, d], dtype=torch.float16, device="cuda")
    K = torch.randn([b, h, s, d], dtype=torch.float16, device="cuda")
    V = torch.randn([b, h, s, d], dtype=torch.float16, device="cuda")

    O_cuda = fmha_cu.ops.fmha(Q, K, V)

    O_ref = F.scaled_dot_product_attention(Q, K, V)

    torch.testing.assert_close(O_cuda, O_ref, atol=5e-4, rtol=0)
