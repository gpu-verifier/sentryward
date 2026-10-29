"""Mathematical reference; source (pinned):
https://github.com/vllm-project/vllm/blob/605c3ddcbaaee06a2687a961ae7be51aec5dddc0/csrc/libtorch_stable/dsv3_fused_a_gemm.cu

fused_a_gemm_kernel<1, gemm_m = 1536 (hd_out), gemm_k = 7168 (hd_in), tile_m 16, tile_n 8, tile_k 256, 16 stages>
(output, mat_a, mat_b, gemm_n = num_tokens). invokeFusedAGemm swaps the operands: the kernel's mat_a is the weight
(vLLM mat_b, a packed column-major [hd_in, hd_out] tensor, i.e. [hd_out, hd_in] row-major) and its mat_b is the
activation ([num_tokens, hd_in] row-major). The epilogue stores gmem_c[n * gemm_m + m] = bf16(acc) for n < gemm_n:
  output[t, o] = bf16(sum_k activation[t, k] * weight[o, k])
"""
import torch


def reference(weight, activation):
    return (activation.float() @ weight.float().T).to(torch.bfloat16)
