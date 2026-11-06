# Source: https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cuda/MaxUnpooling.cu
# Source: https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cpu/MaxUnpoolKernel.cpp
"""max_unpooling2d_forward_kernel<float> (MaxUnpooling.cu L30-L50): per-plane scatter.

Each input element i stores output[n, c, maxind] = input[n, c, i] with maxind = (int) indices[n, c, i]
(MaxUnpooling.cu L46-L48); the CPU kernel cpu_max_unpool performs the same store for ascending i
(MaxUnpoolKernel.cpp L68-L77). Cells that no index selects keep their initial contents (the host zero-fills output
before the launch, MaxUnpooling.cu L182; the kernel itself only scatters). The loop is that scatter for ascending i,
a later store replacing an earlier one. Defined where the selected slots are distinct within a plane; duplicates are
an order-dependent race.
"""
import torch


def reference(input, indices, output):
    n, c, ih, iw = input.shape
    oh, ow = output.shape[2], output.shape[3]
    plane_in = ih * iw
    positions = torch.arange(oh * ow, device=input.device).reshape(1, 1, oh, ow)
    # `int maxind = indices[linearIndex]` keeps the low 32 bits; the asserted range is [0, oH*oW).
    slots = (indices & 0xFFFFFFFF).reshape(n, c, plane_in)
    values = input.reshape(n, c, plane_in)
    result = output
    for i in range(plane_in):
        hit = positions == slots[:, :, i].reshape(n, c, 1, 1)
        result = torch.where(hit, values[:, :, i].reshape(n, c, 1, 1), result)
    return result
