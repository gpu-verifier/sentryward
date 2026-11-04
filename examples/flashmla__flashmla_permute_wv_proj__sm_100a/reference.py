"""FlashMLA permute_wv_proj: per-row 32-channel chunk transpose of wv_proj and of its ue8m0 scale bytes.

Source (FlashMLA 395cb3afbb331fef7b1b21e9340861b9436b42d8):
  csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/kernel.cu:28-30
    (CHUNK_SIZE = INPUT_GRAN = 32, D_O = 512), :50-58 (row (group, r) of wv_group_size*D_O bytes loaded),
    :61-70 (scale byte i = head*16 + chunk goes to byte head + chunk*wv_group_size; byte j of a row sits in
    int32 word j/4, byte j%4), :74-83 (chunk i = head*16 + c goes to chunk c*wv_group_size + head), :88-94 (store);
  flash_mla/fused_norm_rope_attn_rope_cast.py:185-207 (contract).
Weights and scale factors are passed as raw bytes (uint8 views): the kernel moves bytes.
sf_bytes is [n_wv_group, d_proj_out, words, 4]: byte k of int32 word w holds scale index 4*w + k.
"""
import torch


def reference(wv_proj, sf_bytes, wv_group_size):
    g, rows, width = wv_proj.shape
    chunks = width // wv_group_size // 32                                     # 16 chunks of 32 per head
    w = wv_proj.reshape(g, rows, wv_group_size, chunks, 32).transpose(2, 3).reshape(g, rows, width)
    words = sf_bytes.shape[2]
    sf = sf_bytes.reshape(g, rows, wv_group_size, chunks).transpose(2, 3).reshape(g, rows, words, 4)
    return w, sf
