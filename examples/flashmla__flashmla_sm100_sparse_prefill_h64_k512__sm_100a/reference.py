"""FlashMLA SM100 sparse prefill head64 (phase1_h64_k512), bf16 KV, no attention sink, no topk_length
(O, max_logits and LSE).

Source (FlashMLA 395cb3afbb331fef7b1b21e9340861b9436b42d8):
  csrc/cuda_kernels/sm100/prefill/sparse/fwd/head64/config.h (MMA_M = h_q = 64, B_TOPK = 64, D_V = 512) and
    phase1.cuh:175-180 (num_k_blocks = max(ceil(topk_length, 64), 2)), csrc/cuda_kernels/sm100/common_subroutine.h:14-44
    (a token is valid iff 0 <= index < s_kv and its top-k position < topk_length), phase1.cuh:216-299 (masked exp2
    softmax with sm_scale_div_log2, running row max, O rescale), :303-330 (empty row: li = 0, mi = -inf;
    max_logits = real_mi*ln2; LSE = mi*ln2 + ln(li), -inf -> +inf), :334-390 (O = acc / (li + 2**(sink*log2e - mi)),
    no sink -> acc / li; no valid index -> O = 0);
  csrc/api/sparse_prefill.cpp (host contract), tests/ref.py:19-57 (ref_sparse_attn_fwd: indices < 0 or >= s_kv are
  invalid; max_logits = max(P*sm_scale); LSE without the sink; lonely rows: LSE = +inf, O = 0).
The kernel's running maximum mi cancels in exact arithmetic, so the softmax is written here without it.
Scores use the stored sm_scale_div_log2 scalar with exp2, exactly as the kernel does; max_logits uses the f32 value
of CUDART_LN2_F as the kernel does.
Q, KV and indices are strided views (tests/kernelkit non_contiguousify): the reference receives their whole
contiguous storage tensors and slices the logical blocks with the kernel's h_q and topk parameters (h_kv = 1).
"""
import torch

LN2 = 0.6931471824645996  # CUDART_LN2_F (math_constants.h: 6.931471805e-1F) as float32


def reference(Q, KV, indices, sm_scale_log2, h_q, topk):
    # Q: bf16 storage [s_q, h_q+1, 640], logical block [s_q, h_q, 512].  KV: bf16 storage [s_kv, 2, 640], logical
    # block [s_kv, h_kv=1, 512].  indices: int32 storage [s_q, 2, topk+128], logical block [s_q, h_kv=1, topk].
    # sm_scale_log2 = sm_scale * log2(e).
    Q = Q[:, :h_q, :512].contiguous()
    KV = KV[:, :1, :512].contiguous()
    indices = indices[:, :1, :topk].contiguous()
    s_q, h_q, d_qk = Q.shape
    s_kv = KV.shape[0]
    idx = indices.reshape(s_q, topk)
    valid = (idx >= 0) & (idx < s_kv)                                        # [s_q, topk]
    rows = torch.where(valid, idx, 0).reshape(-1).long()
    gathered = KV.reshape(s_kv, d_qk).index_select(0, rows).reshape(s_q, topk, d_qk).float()
    score = (Q.float() @ gathered.transpose(-1, -2)) * sm_scale_log2         # [s_q, h_q, topk], log2 units
    masked = torch.where(valid.unsqueeze(1), score, float("-inf"))
    max_logits = masked.amax(dim=-1) * LN2                                   # [s_q, h_q]
    weight = torch.where(valid.unsqueeze(1), torch.exp2(score), 0.0)
    total = weight.sum(dim=-1)                                               # [s_q, h_q]
    scale = torch.where(total == 0.0, 0.0, torch.reciprocal(total))         # V = the whole 512-dim token
    out = (weight @ gathered) * scale.unsqueeze(-1)
    lse = torch.where(total == 0.0, float("inf"), torch.log(total))
    return out.to(torch.bfloat16), max_logits, lse
