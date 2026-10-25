# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L3174-L3310
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L3564-L3843
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L3983-L4171
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/mask.py#L893-L920
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/interface.py#L2124-L2196
# Source: https://github.com/NVIDIA/cutlass/blob/v4.6.2/include/cute/atom/mma_traits_sm100_frag.hpp#L168-L196
"""FlashAttention-4 SM100 backward main kernel, fp16, d=128, causal, GQA 4:1, 2-CTA.

Fixed scenario (examples/<kernel>/input.json): b=1, h_q=4, h_kv=1, seqlen_q=128, seqlen_k=256,
d=dv=128, cluster 2 (use_2cta_instrs), not deterministic.  Because qhead_per_kvhead > 1 the
kernel accumulates dK and dV into fp32 dKaccum/dVaccum (dKV_postprocess) instead of writing
dK/dV; all three accumulators are read-modify-write (TMA bulk reduce-add), so their prior
contents are inputs.  dQ and dK are unscaled here (softmax_scale is applied by postprocess).

The accumulators use the kernel-internal flat layouts that flash_bwd_postprocess consumes; the
reference returns them in memory order (see _pack_dq / _pack_dkv).
"""
import torch


def _attention_backward_gqa(Q, K, V, dO, LSE, dPsum, scale_log2, causal):
    batch, heads, rows, dim = Q.shape
    kv_heads, columns, dim_v = K.shape[1], K.shape[2], V.shape[3]
    group = heads // kv_heads
    query = torch.arange(rows, device=Q.device).reshape(1, rows, 1)
    key = torch.arange(columns, device=Q.device).reshape(1, 1, columns)
    dqs, dks, dvs = [], [], []
    for b in range(batch):
        for g in range(kv_heads):
            # All query heads sharing kv head g are stacked along rows: one 2-D product per kv head,
            # and the reductions over (query head, row) give the GQA sums for dK and dV.
            q = Q[b, g * group:(g + 1) * group].float().reshape(group * rows, dim)
            do = dO[b, g * group:(g + 1) * group].float().reshape(group * rows, dim_v)
            lse = LSE[b, g * group:(g + 1) * group].reshape(group * rows, 1)
            dpsum = dPsum[b, g * group:(g + 1) * group].reshape(group * rows, 1)
            k = K[b, g].float()
            v = V[b, g].float()
            # P = exp2(S * scale_log2 - LSE_log2) (flash_bwd_sm100.py compute_loop).
            logits = (q @ k.T) * scale_log2 - lse
            if causal:
                # Bottom-right aligned causal mask: key <= query + seqlen_k - seqlen_q (mask.py).
                logits = logits.reshape(group, rows, columns)
                logits = torch.where(key <= query + (columns - rows), logits, -float("inf"))
                logits = logits.reshape(group * rows, columns)
            probability = torch.exp2(logits)
            # dS = P * (dP - dPsum), dP = dO V^T; P and dS are rounded to the MMA input dtype.
            derivative = probability * ((do @ v.T) - dpsum)
            p = probability.to(Q.dtype).float()
            ds = derivative.to(Q.dtype).float()
            dqs.append((ds @ k).reshape(group, rows, dim))
            dks.append(ds.T @ q)
            dvs.append(p.T @ do)
    dq = torch.stack(dqs).reshape(batch, heads, rows, dim)
    dk = torch.stack(dks).reshape(batch, kv_heads, columns, dim)
    dv = torch.stack(dvs).reshape(batch, kv_heads, columns, dim_v)
    return dq, dk, dv


def _pack_dq(dq, ncol):
    # 2-CTA dQaccum tile layout (per batch, head, 128-row m_block):
    #   row = 64*v + m (v = CTA rank, m < 64), col = 64*hh + ncol*s + 4*i + j
    #   offset = 8192*v + 128*ncol*s + 512*i + 256*hh + 4*m + j   (fp32 elements)
    batch, heads, rows, dim = dq.shape
    half = dim // 2
    x = dq.reshape(batch, heads, rows // 128, 2, 64, 2, half // ncol, ncol // 4, 4)
    return x.permute(0, 1, 2, 3, 6, 7, 5, 4, 8)


def _pack_dkv(dk):
    # 2-CTA dKaccum/dVaccum tile layout (per batch, kv head, 128-key n_block):
    #   key = r, col = 64*wg + 32*e + 4*i + j
    #   offset = 8192*wg + 4096*e + 512*i + 4*r + j   (fp32 elements)
    batch, heads, columns, dim = dk.shape
    x = dk.reshape(batch, heads, columns // 128, 128, 2, dim // 64, 8, 4)
    return x.permute(0, 1, 2, 4, 5, 6, 3, 7)


def reference(Q, K, V, dO, LSE, dPsum, dQaccum, dKaccum, dVaccum, scale_log2):
    dq, dk, dv = _attention_backward_gqa(Q, K, V, dO, LSE, dPsum, scale_log2, causal=True)
    # Non-deterministic 2-CTA hdim-128 path: dQ_reduce_ncol = 8 (flash_bwd_sm100.py L249-L252).
    return (dQaccum + _pack_dq(dq, 8),
            dKaccum + _pack_dkv(dk),
            dVaccum + _pack_dkv(dv))
