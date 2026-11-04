# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L240-L259
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_sm100.py#L3590-L3790
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_bwd_postprocess.py#L180-L188
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/interface.py#L1940-L1965
"""FlashAttention-4 SM100 backward, dense, bf16, d=64, 1-CTA (head_dim < 128), non-deterministic.

P = exp2(S*scale_log2 - LSE); dS = P*(dO V^T - dPsum); P and dS are rounded to bf16.
dQ = dS K (unscaled; the postprocess kernel scales it) is bulk-reduce-added into the fp32 dQaccum
buffer in the kernel's memory order: dQ_reduce_ncol = 32, so each 128-row m_block is reduced as
hdim/32 stages of 128*32 values; the register->smem tiled_copy_1d gives each TMEM lane (row m) four
consecutive columns per 512-element chunk. Per tile: offset = 4096*s + 512*i + 4*m + j with
col = 32*s + 4*i + j, i.e. offset = 512*(col//4) + 4*m + col%4 (fp32 elements).
dK = softmax_scale * dS^T Q and dV = P^T dO are stored as bf16.
"""
import torch


def _pack_dq_rows(dq_rows):
    # (rows, d) -> dQaccum memory-order view (rows//128, d//4, 128, 4)
    rows, dim = dq_rows.shape
    return dq_rows.reshape(rows // 128, 128, dim // 4, 4).permute(0, 2, 1, 3)


def reference(Q, K, V, dO, LSE, dPsum, dQaccum, scale, scale_log2):
    probability = torch.exp2((Q.float() @ K.float().T) * scale_log2 - LSE[:, None])
    derivative = probability * ((dO.float() @ V.float().T) - dPsum[:, None])
    p = probability.to(Q.dtype).float()
    ds = derivative.to(Q.dtype).float()
    dq = dQaccum + _pack_dq_rows(ds @ K.float())
    dk = ((ds.T @ Q.float()) * scale).to(Q.dtype)
    dv = (p.T @ dO.float()).to(Q.dtype)
    return dq, dk, dv
