import torch
from examples._reference.pytorch_blockscaled import product


def reference(A, B, scale_a, scale_b):
    result = product(A, B, scale_a, scale_b, packed=False, block=32, e8m0=True)
    return result.to(torch.float32)
