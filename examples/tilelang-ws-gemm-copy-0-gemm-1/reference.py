"""One CTA's 128 x 128 output tile of the compiled 1024 x 1024 GEMM."""


def reference(A, B):
    # Upstream uses A[M, K] @ B[K, N], with separate output C.
    # Full input allocations retain their original contiguous row strides.
    return (A[:128, :].float() @ B[:, :128].float()).to(A.dtype)
