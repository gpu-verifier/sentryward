# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/prefill.cuh#L1673-L1716
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/prefill.cuh#L1785-L1830
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/prefill.cuh#L2321-L2742
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/prefill.cuh#L2780-L2978
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/variants.cuh#L32-L93
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/variant_helper.cuh#L85-L89
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/csrc/single_prefill.cu#L35-L111
"""Single-request FA2 prefill attention, MaskMode::kCausal, DefaultAttention<no custom mask,
no window, no softcap, no alibi>, PosEncodingMode::kNone, bf16 q/k/v/o, f32 QK accumulation, NHD.

q: [qo_len, num_qo_heads, head_dim]; k, v: [kv_len, num_kv_heads, head_dim];
o: [qo_len, num_qo_heads, head_dim] (o_stride_n = num_qo_heads * head_dim, o_stride_h = head_dim).
qo_head = kv_head * group_size + r.  Causal mask (logits_mask): key j is kept iff
j + qo_len <= kv_len + i.

    o[i, h] = sum_{j kept} 2^(scale_log2 * <q[i,h], k[j,kvh]>) v[j,kvh] / sum_{j kept} 2^(...)

scale_log2 = DefaultAttention::sm_scale_log2 = float(double(sm_scale) * log2e_f32)
(PTX: mul.f64 by 0d3FF7154760000000, cvt.rn.f32.f64).  The running maximum cancels in ideal-real
arithmetic, as in the corpus attention references.
"""
import struct

import torch


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


_LOG2E_F32 = _f32(1.44269504088896340736)  # math::log2e (float)


def reference(q, k, v, sm_scale):
    scale_log2 = _f32(sm_scale * _LOG2E_F32)
    qo_len, heads, dim = q.shape
    kv_len, kv_heads, _ = k.shape
    group = heads // kv_heads
    query = q.float().permute(1, 0, 2).reshape(kv_heads, group, qo_len, dim)      # [kvh, g, i, d]
    key = k.float().permute(1, 2, 0).reshape(kv_heads, 1, dim, kv_len)            # [kvh, 1, d, j]
    value = v.float().permute(1, 0, 2).reshape(kv_heads, 1, kv_len, dim)          # [kvh, 1, j, d]
    score = (query @ key) * scale_log2                                            # [kvh, g, i, j]
    row = torch.arange(qo_len, device=q.device).reshape(qo_len, 1)
    column = torch.arange(kv_len, device=q.device).reshape(1, kv_len)
    score = torch.where(column + qo_len <= row + kv_len, score, -float("inf"))
    weight = torch.exp2(score)
    total = weight.sum(dim=-1, keepdim=True)
    out = (weight @ value) / total                                                # [kvh, g, i, d]
    return out.reshape(heads, qo_len, dim).permute(1, 0, 2).to(torch.bfloat16)
