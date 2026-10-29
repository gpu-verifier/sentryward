# Source: https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/native/cuda/Copy.cu
"""direct_copy_kernel_cuda float identity (Copy.cu L213-L253) on the non-contiguous OffsetCalculator path.

out[r, c] = in[r, c] for the dense [2, 3] input; the output is a column slice with row pitch 16 B
(layout mapped in reference_binding.json).
"""


def reference(x):
    return x.clone()
