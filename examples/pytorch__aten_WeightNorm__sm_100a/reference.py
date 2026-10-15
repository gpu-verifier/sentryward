# Source: https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cuda/WeightNorm.cu ; https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/cuda/DeviceUtils.cuh
from examples._reference.pytorch_models import weight_norm


def reference(v, g):
    return weight_norm(v, g)
