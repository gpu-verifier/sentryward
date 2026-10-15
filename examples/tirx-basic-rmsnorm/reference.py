"""Nonempty FP16 RMSNorm; parameter order is checked against the saved PTX."""
import torch

def reference(X, weight):
    x = X.float()
    return (x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * weight.float()).to(X.dtype)
