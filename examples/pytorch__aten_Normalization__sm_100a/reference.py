# Source: https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cuda/Normalization.cu ; https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cuda/Normalization.cuh ; https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/torch/headeronly/core/TensorAccessor.h
from examples._reference.pytorch_models import batch_statistics


def reference(input, epsilon):
    return batch_statistics(input, epsilon)
