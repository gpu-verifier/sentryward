# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/norm.cuh ; https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/csrc/norm.cu
"""FlashInfer FusedAddRMSNormKernel<8, bf16> (norm.cuh L413-L504), in place on input and residual.

x = float(input) + float(residual) stays in float for the normalization; residual receives bf16(x)
and input receives x * rsqrt(sum(x^2) / d + eps) * (weight_bias + float(weight)).
"""
import torch


def reference(input, residual, weight, weight_bias, eps):
    x = input.float() + residual.float()
    rms_rcp = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    normalized = x * rms_rcp * (weight_bias + weight.float())
    return normalized.to(input.dtype), x.to(residual.dtype)
