# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_sm100.py#L426-L450
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_sm100.py#L475-L498
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_sm100.py#L2296-L2368
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_sm100.py#L3068-L3145
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/seqlen_info.py#L67-L136
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/mask.py#L615-L672
import torch


def reference(Q, K, V, cu_seqlens_q, cu_seqlens_k, scale_log2):
    # Packed varlen layout: Q/O are (total_q, head, dim), K/V are (total_k, head, dim).
    total_q, total_k = Q.shape[0], K.shape[0]
    query = torch.arange(total_q, dtype=torch.int32, device=Q.device).reshape(total_q, 1)
    key = torch.arange(total_k, dtype=torch.int32, device=Q.device).reshape(1, total_k)
    # A query token attends to every key of its own sequence (non-causal).
    allowed = None
    for b in range(cu_seqlens_q.shape[0] - 1):
        same = ((query >= cu_seqlens_q[b]) & (query < cu_seqlens_q[b + 1])
                & (key >= cu_seqlens_k[b]) & (key < cu_seqlens_k[b + 1]))
        allowed = same if allowed is None else allowed | same
    q = Q.float().permute(1, 0, 2)
    k = K.float().permute(1, 0, 2)
    v = V.float().permute(1, 0, 2)
    score = (q @ k.transpose(-1, -2)) * scale_log2
    # Common row scaling (running max) cancels under ideal-real semantics.
    weight = torch.where(allowed, torch.exp2(score), 0.0)
    total = weight.sum(dim=-1, keepdim=True)
    output = (weight @ v) / torch.where(total > 0, total, 1.0)
    return output.permute(1, 0, 2).to(torch.bfloat16)
