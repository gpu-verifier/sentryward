# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/decode.cuh#L63-L114
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/decode.cuh#L218-L376
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/decode.cuh#L670-L711
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/variants.cuh#L32-L93
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/variant_helper.cuh#L85-L89
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/csrc/single_decode.cu#L36-L104
"""Single-request decode attention, DefaultAttention<no mask, no window, no softcap, no alibi>,
PosEncodingMode::kNone, bf16 q/k/v/o, NHD KV layout.

q: [num_qo_heads, head_dim]; k, v: [kv_len, num_kv_heads, head_dim]; o: [num_qo_heads, head_dim].
kv_head = qo_head // group_size (group_size = num_qo_heads / num_kv_heads = 1 here).

    o[h] = sum_n 2^(scale_log2 * <q[h], k[n, h]>) v[n, h] / sum_n 2^(scale_log2 * <q[h], k[n, h]>)

scale_log2 is the kernel's DefaultAttention::sm_scale_log2 = float(double(sm_scale) * log2e_f32)
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
    heads, dim = q.shape
    kv_len, kv_heads, _ = k.shape
    group = heads // kv_heads
    query = q.float().reshape(kv_heads, group, dim)            # [kvh, g, d]
    key = k.float().permute(1, 2, 0).reshape(kv_heads, 1, dim, kv_len)   # [kvh, 1, d, n]
    value = v.float().permute(1, 0, 2).reshape(kv_heads, 1, kv_len, dim)  # [kvh, 1, n, d]
    score = (query.unsqueeze(2) @ key) * scale_log2           # [kvh, g, 1, n]
    weight = torch.exp2(score)
    total = weight.sum(dim=-1, keepdim=True)
    out = (weight @ value) / total                            # [kvh, g, 1, d]
    return out.reshape(heads, dim).to(torch.bfloat16)
