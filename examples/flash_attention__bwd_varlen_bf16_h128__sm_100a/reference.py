# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L3174-L3310
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L3564-L3843
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L3846-L3981
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L1780-L1820
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/seqlen_info.py#L67-L140
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/mask.py#L888-L892
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/interface.py#L2143-L2156
# Source: https://github.com/NVIDIA/cutlass/blob/v4.6.2/include/cute/atom/mma_traits_sm100_frag.hpp#L168-L196
"""FlashAttention-4 SM100 backward, packed varlen, bf16, d=128, h=hkv=8 (MHA), non-causal, 2-CTA.

Scenario (input.json): batch 2, cu_seqlens_q = [0,128,256], cu_seqlens_k = [0,160,320]
(shape_arrays), corpus recipe dense_bwd(bf16, d=128, varlen=True) with causal=False.
- Tokens are packed (t, h, d).  cu_seqlens are routing inputs: the same-sequence mask, the batch
  and local position of every token and its padded LSE/dPsum/dQaccum row are computed from them.
- P = exp2(S*scale_log2 - LSE_log2[h, padded_row(q)]) on same-sequence pairs (mask_seqlen only).
- dQ (unscaled) is bulk-reduce-added into dQaccum at padded rows
  padded_offset_q(b) = (cu_q[b] + 128*b) // 128 * 128, in the 2-CTA tile layout (dQ_reduce_ncol 8).
  Padding rows are not written (old contents returned).
- qhead_per_kvhead == 1 with cu_seqlens_k => use_tma_store False => epilogue_dKV: dV = P^T dO and
  dK = softmax_scale * dS^T Q are stored as bf16 (overwrite) for every key < seqlen_k of its batch.
"""
import torch


def _pack_dq_rows(dq_rows, ncol):
    # dq_rows: (h, padded_rows, d) -> 2-CTA dQaccum memory-order view per 128-row tile:
    #   row = 64*v + m, col = 64*hh + ncol*s + 4*i + j
    #   offset = 8192*v + 128*ncol*s + 512*i + 256*hh + 4*m + j   (fp32 elements)
    heads, rows, dim = dq_rows.shape
    half = dim // 2
    x = dq_rows.reshape(heads, rows // 128, 2, 64, 2, half // ncol, ncol // 4, 4)
    return x.permute(0, 1, 2, 5, 6, 4, 3, 7)


def reference(Q, K, V, dO, LSE, dPsum, dQaccum, cu_seqlens_q, cu_seqlens_k,
              scheduler_cu_seqlens_k, softmax_scale, scale_log2):
    # scheduler_cu_seqlens_k is the tile scheduler's copy of cu_seqlens_k (a fixed shape-array
    # table); it only drives work distribution and is bound for the fixed-input context.
    total_q, heads, dim = Q.shape
    total_k = K.shape[0]
    padded_rows = LSE.shape[1]
    device = Q.device
    tq = torch.arange(total_q, device=device)
    tk = torch.arange(total_k, device=device)
    cu_q = cu_seqlens_q.long()
    cu_k = cu_seqlens_k.long()
    # Batch of each packed token = number of sequence ends at or before it.
    bq = (tq.reshape(-1, 1) >= cu_q[1:].reshape(1, -1)).to(torch.int64).sum(dim=-1)
    bk = (tk.reshape(-1, 1) >= cu_k[1:].reshape(1, -1)).to(torch.int64).sum(dim=-1)
    q_start = cu_q.index_select(0, bq)
    same_sequence = bq.reshape(-1, 1) == bk.reshape(1, -1)
    # Padded LSE / dPsum / dQaccum row of each query token (seqlen_info.padded_offset_q).
    padded_row = (tq - q_start) + torch.floor_divide(q_start + 128 * bq, 128) * 128
    rows = torch.arange(padded_rows, device=device)
    scatter = torch.where(rows.reshape(-1, 1) == padded_row.reshape(1, -1), 1.0, 0.0)
    dq_heads, dk_heads, dv_heads = [], [], []
    for h in range(heads):
        q, k, v, do = Q[:, h].float(), K[:, h].float(), V[:, h].float(), dO[:, h].float()
        lse = LSE[h].index_select(0, padded_row).reshape(-1, 1)
        dpsum = dPsum[h].index_select(0, padded_row).reshape(-1, 1)
        logits = torch.where(same_sequence, (q @ k.T) * scale_log2 - lse, -float("inf"))
        probability = torch.exp2(logits)
        derivative = probability * ((do @ v.T) - dpsum)
        p = probability.to(Q.dtype).float()
        ds = derivative.to(Q.dtype).float()
        dq_heads.append(scatter @ (ds @ k))          # (padded_rows, d); zero on padding rows
        dk_heads.append(((ds.T @ q) * softmax_scale).to(Q.dtype))
        dv_heads.append((p.T @ do).to(Q.dtype))
    dq_rows = torch.stack(dq_heads)                   # (h, padded_rows, d)
    dK = torch.stack(dk_heads, dim=1)                 # (total_k, h, d)
    dV = torch.stack(dv_heads, dim=1)
    # Non-deterministic 2-CTA hdim-128 path: dQ_reduce_ncol = 8 (flash_bwd_sm100.py L249-L252).
    return dQaccum + _pack_dq_rows(dq_rows, 8), dK, dV
