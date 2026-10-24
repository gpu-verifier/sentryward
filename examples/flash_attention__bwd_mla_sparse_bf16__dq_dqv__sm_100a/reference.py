# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_mla_dq_dqv_sm100.py#L3-L30
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_mla_dq_dqv_sm100.py#L778-L835
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/topk_gather_kv.py#L195-L278
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/interface.py#L2879-L2881
import torch


def reference(dS, K, V, I, seqlen_k):
    # dS (b, s_q, heads, topk) bf16, K (b, s_k, d) bf16, V (b, s_k, dv) bf16, I (b, s_q, topk) int32.
    batch, seqlen_q, _, topk = dS.shape
    rows = K.shape[1]
    # Rows with I outside [0, seqlen_k) are not loaded; cp.async zero-fills them (src-size operand).
    valid = (I >= 0) & (I < seqlen_k)
    safe = torch.where(valid, I, 0).long()
    base = (torch.arange(batch, device=I.device) * rows).reshape(batch, 1, 1)
    flat = (base + safe).reshape(-1)
    def gather(cache):
        selected = cache.reshape(batch * rows, cache.shape[-1]).index_select(0, flat)
        selected = selected.reshape(batch, seqlen_q, topk, cache.shape[-1]).float()
        return torch.where(valid.unsqueeze(-1), selected, 0.0)
    ds = dS.float()
    dQ = ds @ gather(K)
    dQv = ds @ gather(V)
    return dQ.to(torch.bfloat16), dQv.to(torch.bfloat16)
