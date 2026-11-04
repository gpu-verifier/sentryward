# Source: https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/flash_fwd_sm100.py
# Source: https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/interface.py
# Source: https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/tile_scheduler.py
# Source: https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/testing.py
# Source: https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/tests/cute/test_flash_attn.py
# Upstream semantics: tests/cute/test_flash_attn.py::test_flash_attn_output, first hd256 row of the (255, 256) pair
# ("SM100 hd256 2CTA test case"; seqlen_q = 255, seqlen_k = 256, d = dv = 256, MHA 6/6 heads, batch 9, bf16,
# causal=False, no window,
# softcap 0, no learnable sink). flash_attn_func(q, k, v) with requires_grad inputs returns
# out = softmax(softmax_scale * Q K^T) V and the natural-log LSE (interface._flash_attn_fwd allocates
# lse when an input requires grad), with softmax_scale = 1/sqrt(head_dim), as in testing.attention_ref.
# The kernel folds log2(e) into the scale (utils.compute_softmax_scale_log2) and evaluates
# exp2(scale_log2 * S - rowmax); it writes LSE = (rowmax * scale_log2 + log2(rowsum)) * ln 2
# (flash_fwd_sm100.correction_loop). Under ideal-real semantics the row-max shift cancels, so
# O = sum_k 2^(scale_log2 * S_k) V_k / sum_k 2^(scale_log2 * S_k) and LSE = ln(sum_k 2^(scale_log2 * S_k)).
import torch


def reference(Q, K, V, scale_log2):
    # Q: (batch, heads, seqlen_q, d); K: (batch, kv_heads, seqlen_k, d); V: (batch, kv_heads, seqlen_k, dv)
    batch, heads, rows, dim = Q.shape
    kv_heads = K.shape[1]
    group = heads // kv_heads
    q = Q.float().reshape(batch, kv_heads, group, rows, dim)
    k = K.float().unsqueeze(2)
    v = V.float().unsqueeze(2)
    score = (q @ k.transpose(-1, -2)) * scale_log2
    weight = torch.exp2(score)
    total = weight.sum(dim=-1, keepdim=True)
    out = ((weight @ v) / total).reshape(batch, heads, rows, V.shape[-1]).to(Q.dtype)
    lse = torch.log(total).reshape(batch, heads, rows)
    return out, lse
