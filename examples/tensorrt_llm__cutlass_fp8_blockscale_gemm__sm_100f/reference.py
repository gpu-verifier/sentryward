"""Mathematical reference; sources (pinned):
https://github.com/NVIDIA/TensorRT-LLM/blob/ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/ada_blockwise_gemm/sm89_fp8_gemm_1d1d.cuh
https://github.com/NVIDIA/TensorRT-LLM/blob/ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/ada_blockwise_gemm/sm89_utils.cuh

sm89_fp8_gemm_1d1d_impl<AdaBlockwiseGemmKernel<AdaBlockwiseGemmTraits<e4m3, bf16, float, float, 3, 32, 128, 128>>>:
A is (M, K) row-major, B is (N, K) row-major (K contiguous), D is (M, N) row-major bf16 (gmem_tensor_init and
epilogue_with_smem). Scale granularity (1, 128, 128): scales_a is stored (ScaleM, ScaleK) with stride (1, ScaleM),
ScaleM = M rounded up to 4, i.e. as [ScaleK, ScaleM]; scales_b is stored (ScaleN, ScaleK) with stride (ScaleK, 1).
Every 128-wide K block's partial product is promoted with scale_a[kb, m] * scale_b[n // 128, kb]:
  D[m, n] = sum_kb scale_a[kb, m] * scale_b[n // 128, kb] * sum_{k in kb} A[m, k] * B[n, k]
"""
import torch


def reference(A, B, scale_a, scale_b):
    rows, depth = A.shape
    columns = B.shape[0]
    n_block = torch.arange(columns, device=A.device) // 128
    result = 0.0
    for kb in range(depth // 128):
        lo, hi = kb * 128, (kb + 1) * 128
        product = A[:, lo:hi].float() @ B[:, lo:hi].float().T
        result = result + product * scale_a[kb, :rows][:, None] * scale_b[:, kb][n_block][None, :]
    return result.to(torch.bfloat16)
