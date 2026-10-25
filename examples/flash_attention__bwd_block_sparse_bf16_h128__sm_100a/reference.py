# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L3174-L3310
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L3405-L3470
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L3564-L3843
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/block_sparse_utils.py#L1549-L1611
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/mask.py#L888-L892
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/block_sparsity.py#L415-L438
# Source: https://github.com/NVIDIA/cutlass/blob/v4.6.2/include/cute/atom/mma_traits_sm100_frag.hpp#L168-L196
"""FlashAttention-4 SM100 block-sparse backward, bf16, d=128, MHA, 2-CTA, no mask_mod, non-causal.

Scenario (input.json): b=2, h=8, seqlen_q=seqlen_k=256, sparse block (256, 256)
(q_subtile_factor = kv_subtile_factor = 2).  Backward block-sparse tensors are Q-direction:
for each (b, h, sparse KV block j), cnt[b,h,j] slots of idx[b,h,j,:] name the sparse Q blocks that
the KV tile iterates over (mask list first, then full list; get_m_block_from_iter_bwd).  With no
mask_mod compiled in, mask-list blocks get only the seqlen_k mask, exactly like full-list blocks
(mask.py L888-892).  A sparse Q block listed k times is visited k times, so every contribution
carries that multiplicity.  KV tiles with no visited Q block take the empty-tile path: dK = dV = 0
(flash_bwd_sm100.py L3448-3470), which the zero-weight sum reproduces.

Outputs: dQaccum (fp32, read-modify-write bulk reduce-add, unscaled, 2-CTA tile layout, see
_pack_dq); dK = softmax_scale * sum dS^T Q and dV = sum P^T dO stored as bf16 (TMA store, MHA,
not accumulated).
"""
import torch


def _pack_dq(dq, ncol):
    # 2-CTA dQaccum tile layout (per batch, head, 128-row m_block):
    #   row = 64*v + m (v = CTA rank, m < 64), col = 64*hh + ncol*s + 4*i + j
    #   offset = 8192*v + 128*ncol*s + 512*i + 256*hh + 4*m + j   (fp32 elements)
    batch, heads, rows, dim = dq.shape
    half = dim // 2
    x = dq.reshape(batch, heads, rows // 128, 2, 64, 2, half // ncol, ncol // 4, 4)
    return x.permute(0, 1, 2, 3, 6, 7, 5, 4, 8)


def reference(Q, K, V, dO, LSE, dPsum, dQaccum,
              mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx,
              softmax_scale, scale_log2, sparse_block_q, sparse_block_kv):
    # Q, dO: (b, seqlen_q, h, d); K, V: (b, seqlen_k, h, d); LSE (log2 domain), dPsum: (b, h, seqlen_q).
    batch, rows, heads, dim = Q.shape
    columns = K.shape[1]
    kv_blocks = mask_block_cnt.shape[2]
    query = torch.arange(rows, dtype=torch.int32, device=Q.device).reshape(rows, 1)
    lists = ((mask_block_cnt, mask_block_idx), (full_block_cnt, full_block_idx))
    dq_b, dk_b, dv_b = [], [], []
    for b in range(batch):
        dq_h, dk_h, dv_h = [], [], []
        for h in range(heads):
            q = Q[b, :, h].float()
            do = dO[b, :, h].float()
            lse = LSE[b, h].reshape(rows, 1)
            dpsum = dPsum[b, h].reshape(rows, 1)
            dq = None
            dk_parts, dv_parts = [], []
            for j in range(kv_blocks):
                begin = j * sparse_block_kv
                end = min(columns, begin + sparse_block_kv)
                k = K[b, begin:end, h].float()
                v = V[b, begin:end, h].float()
                # Visit multiplicity of each query row for this sparse KV block (count/index data).
                weight = None
                for count, index in lists:
                    for slot in range(index.shape[-1]):
                        start = index[b, h, j:j + 1, slot:slot + 1] * sparse_block_q      # (1, 1)
                        active = ((count[b, h, j:j + 1].reshape(1, 1) > slot)
                                  & (query >= start) & (query < start + sparse_block_q))  # (rows, 1)
                        term = torch.where(active, 1.0, 0.0)
                        weight = term if weight is None else weight + term
                # P = exp2(S*scale_log2 - LSE); dS = P*(dP - dPsum); both rounded to bf16 for the MMAs.
                probability = torch.exp2((q @ k.T) * scale_log2 - lse)
                derivative = probability * ((do @ v.T) - dpsum)
                p = probability.to(Q.dtype).float() * weight
                ds = derivative.to(Q.dtype).float() * weight
                part = ds @ k
                dq = part if dq is None else dq + part
                dk_parts.append(((ds.T @ q) * softmax_scale).to(Q.dtype))
                dv_parts.append((p.T @ do).to(Q.dtype))
            dq_h.append(dq)
            dk_h.append(torch.cat(dk_parts, dim=0))
            dv_h.append(torch.cat(dv_parts, dim=0))
        dq_b.append(torch.stack(dq_h))           # (h, rows, d)
        dk_b.append(torch.stack(dk_h, dim=1))    # (columns, h, d)
        dv_b.append(torch.stack(dv_h, dim=1))
    dq = torch.stack(dq_b)                        # (b, h, rows, d)
    dK = torch.stack(dk_b)                        # (b, columns, h, d)
    dV = torch.stack(dv_b)
    # Non-deterministic 2-CTA hdim-128 path: dQ_reduce_ncol = 8 (flash_bwd_sm100.py L249-L252).
    return dQaccum + _pack_dq(dq, 8), dK, dV
