"""FlashMLA SM100 head128 sparse decode (small-topk 2-CTA kernel, Decode mode), V4.1 FP8 KV cache, no extra cache,
no split, topk_length, no attention sink (O and LSE).

Source (FlashMLA 395cb3afbb331fef7b1b21e9340861b9436b42d8):
  csrc/cuda_kernels/kv_cache_format.h:5-31 (V4.1 fp8 token = 512 e4m3 bytes + 16 ue8m0 scales = 528 B; each CTA of the
    pair dequantizes its 256-dim half: KVCachePart<V41, 2, cta>);
  csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/config.h (H_Q = 128 over 2 CTAs, B_TOPK = 64,
    D_V = 512) and phase1.cuh:176-198 (topk_length = params.topk_length[batch]; blocks = max(ceil(len,64),64)/64),
    :640-660 (a token is valid iff its index != -1 and its top-k position < topk_length),
    :765-842 (masked exp2 softmax with sm_scale_div_log2 and a running row max, O rescale),
    :844-878 (li == 0 -> O scale 0; no sink -> O scale = 1/li; LSE = mi*ln2 + ln(li), -inf -> +inf);
  csrc/cuda_kernels/sm100/dequant_utils.cuh (fp8 x ue8m0 -> bf16, exact);
  csrc/api/sparse_decode.cpp (SparseDecode_Sm100_Head128_Impl, host contract), tests/ref.py:62-112.
The kernel's running maximum mi cancels in exact arithmetic, so the softmax is written here without it.
Scores use the stored sm_scale_div_log2 scalar with exp2, exactly as the kernel does.
Q and indices are strided views (tests/kernelkit non_contiguousify): the reference receives their whole contiguous
storage tensors and slices the logical block with the kernel's s_q, h_q and topk parameters.
"""
import torch


def _dequant_v41_fp8(data, raw_bytes):
    """[num_blocks, page_block_size, 528] fp8 cache -> [tokens, 512] dequantized keys."""
    num_blocks, page_block_size, token_bytes = data.shape
    tokens = num_blocks * page_block_size
    quant = data.reshape(tokens, token_bytes)[:, :512].float()
    exponent = raw_bytes.reshape(tokens, token_bytes)[:, 512:528].float()
    scale = torch.exp2(exponent - 127.0)  # ue8m0 byte -> 2**(byte-127)
    return (quant.reshape(tokens, 16, 32) * scale.unsqueeze(-1)).reshape(tokens, 512)


def _gather(keys, indices, d_qk):
    b, s_q, topk = indices.shape
    rows = torch.where(indices != -1, indices, 0).reshape(-1).long()
    return keys.index_select(0, rows).reshape(b, s_q, topk, d_qk)


def reference(Q, KV_data, KV_bytes, indices, topk_length, sm_scale_log2, s_q, h_q, topk):
    # Q: bf16 storage [b, s_q+1, h_q+1, 640], logical block [b, s_q, h_q=128, 512].
    # KV: e4m3 and raw-byte views of the cache [num_blocks, page_block_size, 528].
    # indices: int32 storage [b, s_q+1, topk+128], logical block [b, s_q, topk] of token rows (-1 = invalid).
    # topk_length: [b] int32.  sm_scale_log2 = sm_scale * log2(e).  attn_sink is null for this case.
    Q = Q[:, :s_q, :h_q, :512].contiguous()
    indices = indices[:, :s_q, :topk].contiguous()
    b, s_q, h_q, d_qk = Q.shape
    keys = _dequant_v41_fp8(KV_data, KV_bytes)
    gathered = _gather(keys, indices, d_qk)                                  # [b, s_q, topk, 512]
    position = torch.arange(topk, dtype=torch.int32, device=indices.device).reshape(1, 1, topk)
    valid = (indices != -1) & (position < topk_length.reshape(b, 1, 1))      # [b, s_q, topk]
    score = (Q.float() @ gathered.transpose(-1, -2)) * sm_scale_log2         # [b, s_q, h_q, T]
    weight = torch.where(valid.unsqueeze(2), torch.exp2(score), 0.0)
    total = weight.sum(dim=-1)                                               # [b, s_q, h_q]
    scale = torch.where(total == 0.0, 0.0, torch.reciprocal(total))         # V = the whole 512-dim token
    out = (weight @ gathered) * scale.unsqueeze(-1)
    lse = torch.where(total == 0.0, float("inf"), torch.log(total))
    return out.to(torch.bfloat16), lse
