"""CUTLASS example 77 FMHA forward (fp16, NoMask): O = softmax(scale * Q K^T) V and LSE.

Sources (CUTLASS e64a9136dd929639e5f7c969fe5af3bf7415cd4f, examples/77_blackwell_fmha/):
  collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:187-233 (Arguments/Params: scale_softmax = 0 means
    1/sqrt(D); scale_softmax_log2 = log2(e) * scale; scale_output = scale_v * inv_scale_o = 1)
  collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:590-700 (softmax_step: row max of raw S,
    P = exp2(S * scale_log2 - max * scale_log2), row sum), :1092-1140 (O = acc * scale_output / row_sum,
    LSE = log(row_sum) + scale_softmax * row_max)
  77_blackwell_fmha.cu:413-450,625-735 (half_t Q/K/V/O, float LSE, strides, default mainloop arguments)
  reference/fmha_fwd_reference.hpp:58-160 (example reference)
The three scales are the kernel Params at param 0 +896 / +900 / +904, fixed by input.json to the source defaults.
"""
import torch


def reference(q, k, v, scale_softmax, scale_softmax_log2, scale_output):
    s = q.float() @ k.float().T                      # [SQ, SK]
    row_max = s.amax(dim=-1)                         # [SQ]
    p = torch.exp2(s * scale_softmax_log2 - row_max.unsqueeze(-1) * scale_softmax_log2)
    row_sum = p.sum(dim=-1)                          # [SQ]
    o = (p @ v.float()) * scale_output / row_sum.unsqueeze(-1)
    lse = torch.log(row_sum) + scale_softmax * row_max
    return o.to(torch.float16), lse
