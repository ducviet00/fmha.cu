import torch
from torch.nn import functional as F

from fmha import cpu_fmha_fp32

def test_correctness():

    b, h, s, d = 2, 12, 229, 32

    Q = torch.randn([b, h, s, d], dtype=torch.float32, device='cpu')
    K = torch.randn([b, h, s, d], dtype=torch.float32, device='cpu')
    V = torch.randn([b, h, s, d], dtype=torch.float32, device='cpu')
    O = torch.empty([b, h, s, d], dtype=torch.float32, device='cpu')

    cpu_fmha_fp32(Q, K, V, O)

    O_ref = F.scaled_dot_product_attention(Q, K, V)

    assert torch.allclose(O, O_ref, atol=1e-5, rtol=0)