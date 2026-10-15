import torch
from examples._reference.pytorch_blockscaled import product


def reference(A, B, scale_a, scale_b, bias):
    result = product(A, B, scale_a, scale_b, packed=True, block=16, e8m0=False)
    result = result + bias.float()[None, :]
    return result.to(torch.bfloat16)
