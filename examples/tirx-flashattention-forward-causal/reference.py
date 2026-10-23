"""TIRx FlashAttention-4 forward, full launch (128 CTAs): O = softmax(Q K^T / sqrt(128)) V, GQA 8, causal (bottom-right = top-left, S_q = S_kv).

Source: https://github.com/mlc-ai/TIRx-kernels/blob/ff128be7470d674a94169e12c1556590e4046cba/tirx_kernels/ported/flashattention/flash_attention4.py
(make_kernel L98-L1372; tensor maps L1405-L1475; CONFIGS L1538-L1549 s1024_h32kv4; run_test L1607-L1648,
whose arbiter is flash_attn.cute _flash_attn_fwd(softmax_scale=1/sqrt(head_dim), causal=True)).
Q and O are [1, 32, 1024, 128] (batch, query head, position, dim); K and V are [1, 4, 1024, 128].
Query head h uses kv head h // 8 (kv_head_idx * GQA_RATIO + g in the kernel).
The kernel computes exp2((s - m) * scale_log2) with the f32 constant log2(e)/sqrt(128) = 0x3E0293EE,
casts P to f16 for the PV MMA and normalizes by the row sum; floating narrowing is ideal under
ideal_real, so the reference keeps P in f32.
"""
import torch
from examples._reference.pytorch_models import attention


def reference(Q, K, V, scale_log2):
    return attention(Q, K, V, scale_log2, causal=True, splits=1, output_dtype=torch.float16)
