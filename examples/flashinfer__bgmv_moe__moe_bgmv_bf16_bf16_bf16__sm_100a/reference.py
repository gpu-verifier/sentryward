"""Mathematical reference; sources (pinned):
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/bgmv_moe/moe_bgmv_impl.cuh
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/bgmv_moe/moe_bgmv_ops.cu
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/flashinfer/fused_moe/bgmv_moe.py

moe_bgmv_shrink_sliced_kernel<feat_in = 384, feat_out = 8, RANK_TILE 8, 4 pairs per block, 3 stages, bf16, bf16, bf16,
PER_PAIR_INPUT = false> (moe_bgmv_impl.cuh:48-260). The w_ptr table holds, per (slice, expert), a pointer into one
[max_loras, num_experts, rank, feat_in] buffer (fill_w_ptr), and lora_stride = num_experts * rank * feat_in, so
W_base(e, lid) + r*feat_in + i addresses lora_a[lid, e, r, i]. For each pair p < num_pairs (slice 0):
  token = sorted_token_ids[p]; valid iff 0 <= token < num_tokens and lid = lora_indices[token] >= 0
  Y[0, p, r] = bf16(Y[0, p, r] + bf16(sum_i lora_a[lid, expert_ids[p], r, i] * X[token, i] * scale))
Invalid pairs keep Y. Y is read-modify-write: its initial contents are an input.
"""
import torch


def reference(Y, X, lora_a, sorted_token_ids, expert_ids, lora_indices, scale):
    loras, experts = lora_a.shape[0], lora_a.shape[1]
    tokens = X.shape[0]
    token = sorted_token_ids
    token_ok = (token >= 0) & (token < tokens)
    token_c = torch.clamp(token, 0, tokens - 1)
    lid = lora_indices[token_c]
    valid = token_ok & (lid >= 0)
    lid_c = torch.clamp(lid, 0, loras - 1)        # gathers kept in range; no-ops under the input bounds
    expert_c = torch.clamp(expert_ids, 0, experts - 1)
    w = lora_a[lid_c, expert_c]                    # [P, R, F]
    acc = (w.float() * X[token_c].float()[:, None, :] * scale).sum(-1)     # [P, R]
    updated = (Y[0].float() + acc.to(torch.bfloat16).float()).to(torch.bfloat16)
    return torch.where(valid[:, None], updated, Y[0]).unsqueeze(0)
