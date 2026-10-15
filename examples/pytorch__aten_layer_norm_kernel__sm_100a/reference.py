# Source: https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cuda/layer_norm_kernel.cu ; https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cuda/block_reduce.cuh ; https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/SharedReduceOps.h
from examples._reference.pytorch_models import layer_norm_backward


def reference(dY, X, mean, rstd, gamma):
    return layer_norm_backward(dY, X, mean, rstd, gamma)
