import torch
from examples._reference.pytorch_blockscaled import product


def reference(A, B, scale_a, scale_b, alpha):
    result = product(A, B, scale_a, scale_b, packed=True, block=16, e8m0=False)
    result = result * alpha[0]
    return result.to(torch.float16)
