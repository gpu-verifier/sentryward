"""D = A @ B + C (FP16 operands, FP32 accumulator and output) from the upstream matmul_accumulate test (10-tcgen05-copy.py L219-L407)."""
import torch


def reference(A, B, C):
    return (A.float() @ B.float() + C.float()).to(torch.float32)
