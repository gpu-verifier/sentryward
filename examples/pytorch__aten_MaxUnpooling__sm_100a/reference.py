# Source: https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cuda/MaxUnpooling.cu
"""max_unpooling2d_forward_kernel<float> (MaxUnpooling.cu L30-L50): per-plane scatter.

output[n, c, maxind] = input[n, c, i] with maxind = (int) indices[n, c, i]; cells that no index selects keep
their initial contents (the host zero-fills output before the launch; the kernel itself only scatters).
Defined where the selected slots are distinct within a plane; duplicates are an order-dependent race.
"""
import torch


def reference(input, indices, output):
    n, c, ih, iw = input.shape
    oh, ow = output.shape[2], output.shape[3]
    plane_in, plane_out = ih * iw, oh * ow
    values = input.reshape(n, c, 1, plane_in)
    # `int maxind = indices[linearIndex]` keeps the low 32 bits; the asserted range is [0, oH*oW).
    slots = (indices & 0xFFFFFFFF).reshape(n, c, 1, plane_in)
    positions = torch.arange(plane_out, device=input.device).reshape(1, 1, plane_out, 1)
    hit = slots == positions
    scattered = torch.where(hit, values, 0.0).sum(dim=-1)
    written = hit.to(torch.float32).sum(dim=-1) > 0
    result = torch.where(written, scattered, output.reshape(n, c, plane_out))
    return result.reshape(n, c, oh, ow)
