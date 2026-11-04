"""FlashMLA SM100 head64 sparse decode, V4.1 FP8 KV cache, no extra cache, no-split path, topk_length (O and LSE).

Source (FlashMLA 395cb3afbb331fef7b1b21e9340861b9436b42d8):
  csrc/cuda_kernels/kv_cache_format.h:5-19 (V4.1 fp8 token = 512 e4m3 bytes of all 512 dims, RoPE included,
    followed by 16 ue8m0 scale bytes, one per 32 dims; 528 bytes per token);
  csrc/cuda_kernels/sm100/decode/sparse/head64/config.h:38-41 (D_Q = D_K = D_V = 512);
  csrc/cuda_kernels/sm100/dequant_utils.cuh:70-110 and csrc/cuda_kernels/sm100/helpers.h:92-110
    (fp8 x ue8m0 -> bf16, exact: cvt.rn.bf16x2.ue8m0x2 gives 2**(byte-127));
  csrc/cuda_kernels/sm100/decode/sparse/head64/kernel.cuh:121-132 (topk_length = params.topk_length[batch]),
    :563-584 (a token is valid iff its index != -1 and its top-k position < topk_length),
    :199-262 (masked exp2 softmax with sm_scale_div_log2 and a running row max),
    :281-292 (empty row: li = 0), :294-303 (no-split LSE = mi*ln2 + ln(li)),
    :183 (attn_sink scaled by CUDART_L2E_F), :322-352 (O = acc / (li + 2**(sink*log2e - mi)), bf16);
  csrc/api/sparse_decode.cpp (host contract), tests/ref.py:62-112 (ref_sparse_attn_decode: positions >= topk_length
  and index -1 are masked), tests/quant.py (V4.1 fp8 quantize/dequantize).
The kernel's running maximum mi cancels in exact arithmetic, so the softmax is written here without it.
Scores use the stored sm_scale_div_log2 scalar with exp2, exactly as the kernel does.
Q and indices are strided views (tests/kernelkit non_contiguousify): the reference receives their whole contiguous
storage tensors and slices the logical block with the kernel's s_q, h_q and topk parameters.
"""
import torch

L2E = 1.4426950216293335  # CUDART_L2E_F (math_constants.h: 1.442695041F) as float32


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


def reference(Q, KV_data, KV_bytes, indices, topk_length, attn_sink, sm_scale_log2, s_q, h_q, topk):
    # Q: bf16 storage [b, s_q+1, h_q+1, 640], logical block [b, s_q, h_q, 512].  KV: two typed views of the same
    # packed cache bytes [num_blocks, page_block_size, 528]: e4m3 data and raw bytes (for the trailing ue8m0 scales).
    # indices: int32 storage [b, s_q+1, topk+128], logical block [b, s_q, topk] of token rows (-1 = invalid).
    # topk_length: [b] int32.  attn_sink: [h_q] f32 per-head bias.  sm_scale_log2 = sm_scale * log2(e).
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
    sink = torch.exp2(attn_sink.reshape(1, 1, h_q) * L2E)                    # V = the whole 512-dim token
    scale = torch.where(total == 0.0, 0.0, torch.reciprocal(total + sink))
    out = (weight @ gathered) * scale.unsqueeze(-1)
    lse = torch.where(total == 0.0, float("inf"), torch.log(total))
    return out.to(torch.bfloat16), lse
