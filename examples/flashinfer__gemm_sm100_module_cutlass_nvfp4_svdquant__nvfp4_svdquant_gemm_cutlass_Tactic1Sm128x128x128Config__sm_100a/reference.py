import torch
from examples._reference.pytorch_blockscaled import product


def reference(A, B, scale_a, scale_b, low_rank_D, low_rank_L1, alpha, bias):
    result = product(A, B, scale_a, scale_b, packed=True, block=16, e8m0=False)
    result = result + low_rank_D.float() @ low_rank_L1.float().T
    result = result * alpha[0]
    result = result + bias.float()[None, :]
    return result.to(torch.bfloat16)
