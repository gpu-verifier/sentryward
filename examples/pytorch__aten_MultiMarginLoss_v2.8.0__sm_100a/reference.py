# Source: https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/cuda/MultiMarginLoss.cu
"""MultiMarginLoss_backward_kernel<1, float> (v2.8.0, L68-L127): same gradient as the newer sibling.

The v2.8.0 kernel omits the __syncthreads() that later versions place before the gradOutput scaling loop;
this reference states the intended gradient, not the racy schedule.
"""
from examples._reference.pytorch_models import margin_backward


def reference(input, target, gradOutput, margin):
    return margin_backward(input, target, gradOutput, margin)
