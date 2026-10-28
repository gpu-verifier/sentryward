# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/interface.py#L3471-L3473
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_mla_sm100.py#L350-L357
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_mla_sm100.py#L2290-L2380
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_mla_sm100.py#L3041-L3045
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/mask.py#L660-L667
import torch


def reference(Q, Qv, K, V, scale_log2):
    # Q (b, s_q, h, d), Qv (b, s_q, h, dv), K (b, s_k, h_k, d), V (b, s_k, h_k, dv).
    # O = softmax(scale * (Q K^T + Qv V^T)) V, computed with exp2 and scale_log2 = scale * log2(e).
    # Non-causal, dense KV; h == h_k in this scenario (no GQA packing).
    q = Q.float().permute(0, 2, 1, 3)
    qv = Qv.float().permute(0, 2, 1, 3)
    k = K.float().permute(0, 2, 1, 3)
    v = V.float().permute(0, 2, 1, 3)
    score = (q @ k.transpose(-1, -2) + qv @ v.transpose(-1, -2)) * scale_log2
    # The kernel's running-max subtraction is a common row scale; it cancels under ideal-real semantics.
    weight = torch.exp2(score)
    total = weight.sum(dim=-1, keepdim=True)
    out = (weight @ v) / total
    return out.permute(0, 2, 1, 3).to(torch.bfloat16)
