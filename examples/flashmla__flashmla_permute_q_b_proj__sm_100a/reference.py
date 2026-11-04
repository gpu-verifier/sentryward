"""FlashMLA permute_q_b_proj: row permutation of q_b_proj and of its scale factors.

Source (FlashMLA 395cb3afbb331fef7b1b21e9340861b9436b42d8):
  csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/kernel.cu:24-27
    (row r = head*d_q + hd goes to out_row = hd%16 + head*16 + (hd/16)*(h_q*16)),
    :39-46 / :53-61 (the whole row of q_lora_rank fp8 bytes is moved unchanged),
    :48-51 (scale word scale_idx of row r goes to the same word index of out_row);
  flash_mla/fused_norm_rope_attn_rope_cast.py:161-182 (contract).
The weight is passed as its raw bytes (uint8 view of the fp8_e4m3 tensor): the kernel copies bytes.
"""
import torch


def _permute_rows(x, h_q, d_q):
    # [h_q*d_q, n] -> rows reordered as (hd // 16, head, hd % 16)
    n = x.shape[-1]
    return x.reshape(h_q, d_q // 16, 16, n).permute(1, 0, 2, 3).reshape(h_q * d_q, n)


def reference(q_b_proj, scale_factors, h_q, d_q):
    return _permute_rows(q_b_proj, h_q, d_q), _permute_rows(scale_factors, h_q, d_q)
