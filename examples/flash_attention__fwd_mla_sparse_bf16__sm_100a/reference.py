# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/interface.py#L3471-L3478
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_mla_sm100.py#L350-L366
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_mla_sm100.py#L1435-L1450
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_mla_sm100.py#L2290-L2380
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_mla_sm100.py#L3041-L3045
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/topk_gather_kv.py#L164-L278
import torch


def reference(Q, Qv, K, V, topk_indices, scale_log2, seqlen_k):
    # Q (b, s_q, h, d), Qv (b, s_q, h, dv), K (b, s_k, 1, d), V (b, s_k, 1, dv) bf16;
    # topk_indices (b, s_q, topk) int32.  MQA (h_k = 1), non-causal: seqlen_k_limit = seqlen_k.
    # O = softmax(scale * (Q K[I]^T + Qv V[I]^T)) V[I] over the valid top-k slots.
    batch, seqlen_q, _, _ = Q.shape
    rows = K.shape[1]
    topk = topk_indices.shape[-1]
    # Slots outside [0, seqlen_k) are masked to -inf (bitmask) and their K/V rows are not loaded.
    valid = (topk_indices >= 0) & (topk_indices < seqlen_k)
    safe = torch.where(valid, topk_indices, 0).long()
    flat = ((torch.arange(batch, device=Q.device) * rows).reshape(batch, 1, 1) + safe).reshape(-1)
    def gather(cache):
        selected = cache.reshape(batch * rows, cache.shape[-1]).index_select(0, flat)
        selected = selected.reshape(batch, seqlen_q, topk, cache.shape[-1]).float()
        return torch.where(valid.unsqueeze(-1), selected, 0.0)
    k, v = gather(K), gather(V)
    score = (Q.float() @ k.transpose(-1, -2) + Qv.float() @ v.transpose(-1, -2)) * scale_log2
    score = torch.where(valid.unsqueeze(2), score, -float("inf"))
    # The running row-max subtraction is a common row scale; it cancels under ideal-real semantics.
    weight = torch.exp2(score)
    total = weight.sum(dim=-1, keepdim=True)
    # Epilogue: a row whose row_sum is 0 is scaled by 1, so its output is 0.
    out = torch.where(total > 0, (weight @ v) / total, 0.0)
    return out.to(torch.bfloat16)
