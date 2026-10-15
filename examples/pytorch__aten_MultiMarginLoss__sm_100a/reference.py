# Source: https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cuda/MultiMarginLoss.cu
from examples._reference.pytorch_models import margin_backward


def reference(input, target, gradOutput, margin):
    return margin_backward(input, target, gradOutput, margin)
