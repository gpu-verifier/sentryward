"""One CTA's 128 x 128 output tile of the tilelang tcgen05 GEMM (C = A @ B^T).

Source: https://github.com/tile-ai/tilelang/blob/cd37ed5fc35ae7a60a1277c8eb49028174ac51e6/examples/gemm_sm100/gemm_tcgen5mma.py
(L13-L56 kernel; L59-L71 trans_A=False, trans_B=True, bf16 in/out, f32 accumulation,
`ref_c = (a.float() @ b.T.float()).to(bf16)`).
"""
import torch


def reference(A, B):
    # A holds rows 0..127 of A[M, K]; B holds rows 0..127 of B[N, K]; K = 8192.
    # The launch (grid [1,1,1]) computes the (by=0, bx=0) tile only.
    return (A.float() @ B.float().T).to(torch.bfloat16)
