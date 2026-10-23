"""vLLM SM100 CUTLASS MLA decode (paged, page_size 128, split_kv 1, fp16 in/out, ptr_lse null): O only.

Source (vLLM 605c3ddcbaaee06a2687a961ae7be51aec5dddc0, csrc/libtorch_stable/attention/mla/):
  sm100_cutlass_mla_kernel.cu:100-187 (problem (H=128, K=max_seq_len, (512, 64), B); C_latent = cache[..., :512],
    K_rope = cache[..., 512:]; stride_PT = (1, pages_per_seq); O stride (512, 1, H*512); LSE stride (1, H));
  cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp:1091-1180 (page k-tile p of batch b is
    physical page page_table[b, p]), :1620-1660 (mask k >= seq_lens[b] to -inf; exp2(scale*log2e*(s - max))),
    :1829-1870 (O = output_scale * acc / row_sum, ElementOut; LSE only if ptr_lse != nullptr).
V is the latent part of the same cache rows (MLA absorbed form).
"""
import torch

LOG2E_F32 = 1.44269502162933349609375  # static_cast<float>(M_LOG2E), kernel :1649


def reference(q_nope, q_pe, kv_c_and_k_pe_cache, page_table, sequence_lengths, softmax_scale, seq_len):
    # q_nope: [B, H, 512], q_pe: [B, H, 64], cache: [pages, page_size, 576] (fp16);
    # page_table: [B, pages_per_seq] int32 physical page ids (bounded by tensor_properties).
    # sequence_lengths: [B] fixed shape-array tensor, bound only so the fixed-input context finds a common
    # identity; the mask uses seq_len, the same fixed value (/shape_arrays/0/shapes/0/0) as a scalar (B == 1).
    batch, heads, latent = q_nope.shape
    pages, page_size, dim = kv_c_and_k_pe_cache.shape
    pages_per_seq = page_table.shape[1]
    keys = pages_per_seq * page_size
    q = torch.cat((q_nope.float(), q_pe.float()), dim=-1)  # [B, H, 576]
    rows = page_table.reshape(-1).long()
    kv = kv_c_and_k_pe_cache.index_select(0, rows).reshape(batch, keys, dim).float()  # [B, K, 576]
    score = (q @ kv.transpose(-1, -2)) * softmax_scale * LOG2E_F32  # [B, H, K]
    valid = torch.arange(keys, device=q_nope.device).reshape(1, 1, keys) < seq_len
    weight = torch.where(valid, torch.exp2(score), 0.0)
    out = (weight @ kv[..., :latent]) / weight.sum(dim=-1, keepdim=True)  # output_scale = 1
    return out.to(q_nope.dtype)
