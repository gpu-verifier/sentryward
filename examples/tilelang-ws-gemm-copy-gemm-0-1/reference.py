"""Whole 128 x 128 GEMM of the compiled copy_gemm_0_1 example (M=N=128, K=64).

Source: https://github.com/tile-ai/tilelang/blob/cd37ed5fc35ae7a60a1277c8eb49028174ac51e6/examples/warp_specialize/example_warp_specialize_gemm_copy_gemm_0_1.py
(L5-L44 kernel, L47-L71 main configuration and `ref_c = a @ b` check).
"""


def reference(A, B):
    # A[M, K] @ B[K, N] with f32 accumulation (accum_dtype) and f16 output C.
    # The two warpgroups each compute 64 of the 128 output columns.
    return (A.float() @ B.float()).to(A.dtype)
