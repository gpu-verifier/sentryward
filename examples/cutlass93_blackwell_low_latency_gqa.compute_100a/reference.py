"""CUTLASS example 93 TGV low-latency GQA decode (bf16, no sinks).

O[h, :] = sum_l softmax_l(scale_log2-domain scores) V[l, h // group, :] over the valid KV range
[start, seq_len), start = 0 when sliding_window_size == 0 else max(0, seq_len - sliding_window_size).

Sources (CUTLASS e64a9136dd929639e5f7c969fe5af3bf7415cd4f, examples/93_blackwell_low_latency_gqa/):
  tgv_gqa.cuh:1307-1346 (row_valid = kv < seq_len && kv >= seq_len_start; s *= softmax_scale_log2)
  tgv_gqa.cuh:1677-1691 (gqa_device parameters), :1860-1910 (K (kvL,dH,kvH,BS), Q ((qHLocal,qL),dH,kvH,BS),
    V (dH,kvL,kvH,BS), O (dH,(qHLocal,qL),kvH,BS)), :2093-2135 (launch: softmax_scale * Log2_E, NoSink = true)
  tgv_gqa.cu:85-305 (reference_gqa: exp2(s * scale_log2 - max), split-wise rescale, O = acc / fsum), :440-490 (strides)
"""
import torch


def reference(q, k, v, seq_len, scale_log2, sliding_window_size):
    # q [qH, dH] with qH = kvH * qHLocal (qHLocal fastest); k, v [kvL, kvH, dH]; seq_len [BS] (BS = 1).
    kv_len, kv_heads, dim = k.shape
    heads = q.shape[0]
    group = heads // kv_heads
    position = torch.arange(kv_len, device=k.device)
    valid = position < seq_len
    if sliding_window_size != 0:
        valid = valid & (position >= (seq_len - sliding_window_size).clamp_min(0))
    score = q.float().reshape(kv_heads, group, dim) @ k.float().permute(1, 2, 0)   # [kvH, qHLocal, kvL]
    # The kernel subtracts the running row max before exp2; the common factor cancels in O.
    weight = torch.where(valid, torch.exp2(score * scale_log2), 0.0)
    out = (weight @ v.float().permute(1, 0, 2)) / weight.sum(dim=-1, keepdim=True)
    return out.reshape(heads, dim).to(torch.bfloat16)
