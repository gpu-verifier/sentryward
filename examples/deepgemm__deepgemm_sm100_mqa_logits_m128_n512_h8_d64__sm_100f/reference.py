"""DeepGEMM SM100 FP8 MQA logits (contiguous KV, non-compressed logits).

Source (DeepGEMM 559d79fb6994a58b8a15b4b93bf13ccc16edf247):
  deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh:324-336 (FP8 UMMA: accum[n, m*H+h] = sum_d KV[n,d] Q[m,h,d]),
  :408-498 (per-head (a+|a|) * w reduction, /2, * per-KV scale, bf16 store at logits[m*stride + n]);
  deep_gemm/include/deep_gemm/scheduler/sm100_mqa_logits.cuh:42-81 (row/column mapping);
  tests/test_attention.py:84-113 (ref_fp8_mqa_logits: sum_h relu(q.k) * w, masked by [ks, ke)).
The [ks, ke) -inf mask is applied by the separate clean_logits kernel, not by this kernel; in this
scenario ks = 0 and ke = 512 for every row, so every logit is in range and written.
"""
import torch


def reference(Q, KV, SFKV, weights, cu_seq_len_k_start, cu_seq_len_k_end):
    # cu_seq_len_k_start, cu_seq_len_k_end: fixed shape-array tensor(s) bound only so the native fixed-input context finds a
    # common identity (fixed_input_context.hpp decode_verified); values are not used here.
    # Q: [num_q_tokens, num_heads, head_dim] e4m3; KV: [num_kv_tokens, head_dim] e4m3.
    # SFKV: [num_kv_tokens] f32 per-token KV scale; weights: [num_q_tokens, num_heads] f32.
    score = Q.float() @ KV.float().transpose(0, 1)  # [M, H, N]
    # Kernel form of relu(score) * w: (a + |a|) * w summed over heads, then halved.
    reduced = ((score + score.abs()) * weights.float().unsqueeze(-1)).sum(dim=1) / 2  # [M, N]
    return (reduced * SFKV.float().unsqueeze(0)).to(torch.bfloat16)
