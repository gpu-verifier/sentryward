"""Mathematical reference; source (pinned):
https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/blackwell_bgmv_moe/sm100a/blackwell_bgmv_moe_bf16_h2688_sm100a.cu

kernel_flashinfer_bgmv_moe_shrink_bf16_h2688_r32_p4_s3 (L65-287): hidden 2688, rank 32, 4 pairs per block,
rank block of 8 per blockIdx.y. For each pair p < num_pairs:
  token = sorted_token_ids[p], expert = expert_ids[p]; valid iff 0 <= token < num_tokens and
  lora = lora_indices[token] >= 0
  out[p, r] = bf16(valid ? sum_k x[token, k] * lora_a[((lora * num_experts + expert) * 32 + r) * 2688 + k] : 0)
The kernel stores every pair p < num_pairs (invalid pairs get 0.0); the store is not an accumulation.
lora_a is viewed as [num_loras, num_experts, 32, 2688].
"""
import torch


def reference(x, lora_a, sorted_token_ids, expert_ids, lora_indices):
    loras, experts = lora_a.shape[0], lora_a.shape[1]
    tokens = x.shape[0]
    token = sorted_token_ids
    token_ok = (token >= 0) & (token < tokens)
    token_c = torch.clamp(token, 0, tokens - 1)
    lid = lora_indices[token_c]
    valid = token_ok & (lid >= 0)
    lid_c = torch.clamp(lid, 0, loras - 1)
    expert_c = torch.clamp(expert_ids, 0, experts - 1)
    w = lora_a[lid_c, expert_c]                                        # [P, 32, 2688]
    acc = (w.float() * x[token_c].float()[:, None, :]).sum(-1)         # [P, 32]
    return torch.where(valid[:, None], acc, 0.0).to(torch.bfloat16)
