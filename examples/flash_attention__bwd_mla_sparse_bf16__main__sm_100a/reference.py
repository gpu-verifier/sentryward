# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_mla_sm100.py#L160-L166
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_mla_sm100.py#L336-L352
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_mla_sm100.py#L1449-L1460
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_mla_sm100.py#L1879-L1916
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_mla_sm100.py#L2104-L2122
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/topk_gather_kv.py#L195-L278
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/interface.py#L2740-L2876
import torch


def reference(dO, V, Qv, P, scale_P, dPsum, index_topk, dV, softmax_scale, seqlen_k):
    # dO, Qv (b, s_q, h, dv) bf16; V (b, s_k, 1, dv) bf16; P (b, s_q, h, topk) bf16;
    # scale_P (b, s_q, topk // 128, h) fp32; dPsum (b, s_q, h) fp32; index_topk (b, s_q, topk) int32;
    # dV (b, s_k, 1, dv) fp32 read-modify-write accumulator.  Returns (dS, new dV).
    batch, seqlen_q, heads, topk = P.shape
    rows, dv = V.shape[1], V.shape[-1]
    blocks = scale_P.shape[2]
    # V rows outside [0, seqlen_k) are not loaded; cp.async zero-fills them.
    valid = (index_topk >= 0) & (index_topk < seqlen_k)
    safe = torch.where(valid, index_topk, 0).long()
    flat = ((torch.arange(batch, device=P.device) * rows).reshape(batch, 1, 1) + safe).reshape(-1)
    v = V.reshape(batch * rows, dv).index_select(0, flat).reshape(batch, seqlen_q, topk, dv).float()
    v = torch.where(valid.unsqueeze(-1), v, 0.0)
    # P is rescaled per (row, block of topk // blocks columns) by scale_P.
    scale = scale_P.permute(0, 1, 3, 2).unsqueeze(-1)
    p = (P.float().reshape(batch, seqlen_q, heads, blocks, topk // blocks) * scale).reshape(P.shape)
    dp = dO.float() @ v.transpose(-1, -2)
    ds = p * (dp - dPsum.unsqueeze(-1)) * softmax_scale
    update = p.transpose(-1, -2) @ dO.float() + ds.transpose(-1, -2) @ Qv.float()
    # Scatter-add slot rows into dV[b, index, 0, :]; the kernel skips the -1 sentinel, which a
    # one-hot match against arange(s_k) also drops.
    columns = torch.arange(rows, dtype=index_topk.dtype, device=P.device).reshape(1, 1, 1, rows)
    match = torch.where(index_topk.unsqueeze(-1) == columns, 1.0, 0.0)
    match = match.reshape(batch, seqlen_q * topk, rows)
    delta = match.transpose(-1, -2) @ update.reshape(batch, seqlen_q * topk, dv)
    return ds.to(P.dtype), dV + delta.reshape(dV.shape)
