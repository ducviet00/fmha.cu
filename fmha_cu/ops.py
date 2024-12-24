import torch
from torch import Tensor

__all__ = ["fmha"]


def fmha(Q: Tensor, K: Tensor, V: Tensor, op_code: int = 0) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.fmha_cu.fmha.default(Q, K, V, op_code)


@torch.library.register_fake("fmha_cu::fmha")
def _(Q: Tensor, K: Tensor, V: Tensor, op_code: int):
    torch._check(Q.shape == K.shape == V.shape)
    torch._check(Q.dtype == K.dtype == V.dtype)
    torch._check(Q.device == K.device == V.device)
    return torch.empty_like(Q)
