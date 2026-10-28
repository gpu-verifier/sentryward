# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_mla_dk_sm100.py#L77-L90
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_mla_dk_sm100.py#L130-L136
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_mla_dk_sm100.py#L1112-L1128
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/interface.py#L2879-L2886
import torch


def reference(dS, Q, I, dKaccum):
    # dS (b, s_q, heads, topk) bf16, Q (b, s_q, heads, dim) bf16, I (b, s_q, topk) int32,
    # dKaccum (b, seqlen_k, dim) fp32 read-modify-write accumulator.
    batch, seqlen_q, _, topk = dS.shape
    seqlen_k, dim = dKaccum.shape[1:]
    # Per token: dK_t[topk, dim] = dS_t^T @ Q_t (reduction over heads, fp32 accumulate).
    rows = dS.float().transpose(-1, -2) @ Q.float()
    # Scatter-add row k of token t into dKaccum[b, I[b, t, k], :].  The kernel skips the -1
    # sentinel (I < 0); a one-hot match against arange(seqlen_k) drops it as well.
    columns = torch.arange(seqlen_k, dtype=I.dtype, device=I.device).reshape(1, 1, 1, seqlen_k)
    match = torch.where(I.unsqueeze(-1) == columns, 1.0, 0.0)
    match = match.reshape(batch, seqlen_q * topk, seqlen_k)
    delta = match.transpose(-1, -2) @ rows.reshape(batch, seqlen_q * topk, dim)
    return dKaccum + delta
