"""Full FP16 GEMM specification from the upstream Gluon tutorial."""
def reference(A, B):
    return (A.float() @ B.float()).to(A.dtype)
