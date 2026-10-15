"""FP16 GEMM for the upstream M=2000, N=1000, K=2000 test."""
def reference(A, B):
    return (A.float() @ B.float()).to(A.dtype)
