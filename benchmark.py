import torch
from torch.nn import functional as F
from torch.profiler import profile, ProfilerActivity
import triton

import fmha_cu

warmup = 25
rep = 100
b, h, s, d = 2, 12, 4429, 64

Q = torch.randn([b, h, s, d], dtype=torch.float32, device="cuda")
K = torch.randn([b, h, s, d], dtype=torch.float32, device="cuda")
V = torch.randn([b, h, s, d], dtype=torch.float32, device="cuda")

flops_per_matmul =  2.0 * b * h * s * s * d
total_flops = 2 * flops_per_matmul

torch_ms = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(Q, K, V), warmup=warmup, rep=rep)
print(f"Torch SDPA - TFLOPs {total_flops / torch_ms * 1e-9}; ms: {torch_ms}")

op0_ms = triton.testing.do_bench(lambda: fmha_cu.ops.fmha(Q, K, V, op_code=0), warmup=warmup, rep=rep)
print(f"FMHA OP0 - TFLOPs {total_flops / op0_ms * 1e-9}; ms: {op0_ms}")

op1_ms = triton.testing.do_bench(lambda: fmha_cu.ops.fmha(Q, K, V, op_code=1), warmup=warmup, rep=rep)
print(f"FMHA OP1 - TFLOPs {total_flops / op1_ms * 1e-9}; ms: {op1_ms}")


activities = [
    ProfilerActivity.CPU,
    ProfilerActivity.CUDA,
]

with profile(activities=activities, record_shapes=True) as prof:
    for _ in range(50):
        O_ref = F.scaled_dot_product_attention(Q, K, V)
        O_op0 = fmha_cu.ops.fmha(Q, K, V, op_code=0)
        O_op1 = fmha_cu.ops.fmha(Q, K, V, op_code=1)

with open("profile_fmha.txt", "w") as f:
    table = prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", max_name_column_width=40, row_limit=20)
    f.write(table)