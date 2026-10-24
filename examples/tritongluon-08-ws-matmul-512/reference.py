"""Full FP16 GEMM C = A @ B (A is MxK, B is KxN) from the upstream warp-specialized matmul (08-warp-specialization.py L388-L604)."""
def reference(A, B):
    return (A.float() @ B.float()).to(A.dtype)
