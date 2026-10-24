# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/cascade.cuh#L44-L74
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/attention/cascade.cuh#L578-L593
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/csrc/cascade.cu#L23-L58
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/math.cuh#L47-L51
# Source: https://github.com/flashinfer-ai/flashinfer/blob/8c94f70f49fbe17adb5deae5b2b48ff467346863/include/flashinfer/math.cuh#L172-L176
"""MergeStateKernel<8, bf16, bf16>: merge two attention states (v, base-2 LSE).

v_a, v_b: [seq_len, num_heads, head_dim] bf16; s_a, s_b: [seq_len, num_heads] f32
logsumexp values with base 2 (cascade.cuh doc comment).

    m        = max(s_a, s_b)
    w_a      = 2^(s_a - m),  w_b = 2^(s_b - m),  d = w_a + w_b
    v_merged = (w_a / d) * v_a + (w_b / d) * v_b      (d > 0 guard, as the kernel)
    s_merged = log2(d) + m

log2 is written as ln(d) / ln(2) with a symbolic ln(2) (log of the constant 2):
the capture lowering has no aten.log2.  See NOTES.md.
"""
import torch


def reference(v_a, s_a, v_b, s_b):
    sa, sb = s_a.float(), s_b.float()
    m = torch.where(sa > sb, sa, sb)
    wa = torch.exp2(sa - m)
    wb = torch.exp2(sb - m)
    d = wa + wb
    scale_a = torch.where(d > 0, wa / d, 0.0)
    scale_b = torch.where(d > 0, wb / d, 0.0)
    v_merged = scale_a.unsqueeze(-1) * v_a.float() + scale_b.unsqueeze(-1) * v_b.float()
    ln2 = torch.log(torch.ones_like(d) * 2.0)
    s_merged = torch.log(d) / ln2 + m
    return v_merged.to(torch.bfloat16), s_merged
