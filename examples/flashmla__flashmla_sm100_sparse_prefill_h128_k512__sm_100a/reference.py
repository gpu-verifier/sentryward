"""FlashMLA SM100 sparse prefill head128, regular 2-CTA kernel (fwd/head128, phase1_k512), bf16 KV, with attention
sink and topk_length (O, max_logits and LSE).

Source (FlashMLA 395cb3afbb331fef7b1b21e9340861b9436b42d8):
  csrc/cuda_kernels/sm100/prefill/sparse/fwd/head128/config.h (B_H = 128 over 2 CTAs, B_TOPK = 128, D_V = 512) and
    phase1.cuh:87-88 (topk_length = params.topk_length[s_q_idx]; num_k_blocks = max(ceil(topk_length, 128), 1)),
    :589-597 (a token is valid iff 0 <= index < s_kv and its top-k position < topk_length),
    :196-254 (invalid scores -> -inf; exp2 softmax with sm_scale_div_log2 and a running row max mi), :226-289
    (O rescale), :297-316 (no valid token: li = 0, mi = -inf; max_logits = real_mi*ln2; LSE = ln(li) + mi*ln2,
    -inf -> +inf), :323-368 (sink = attn_sink[h]*log2(e); O = acc / (li + 2**(sink - mi)); no valid token in the
    row -> O = 0);
  csrc/api/sparse_prefill.cpp (host contract), flash_mla/flash_mla_interface.py:185-216 (attn_sink: O is multiplied
  by exp(lse) / (exp(lse) + exp(attn_sink)); -inf has no effect, +inf gives O = 0; no effect on lse and max_logits;
  topk_length: query i attends only indices[i, :, :topk_length[i]]), tests/ref.py:19-57 (ref_sparse_attn_fwd:
  positions >= topk_length and indices < 0 or >= s_kv are invalid; max_logits = max(P*sm_scale); LSE without the
  sink; lse_for_o = logsumexp(lse, attn_sink); lonely rows: LSE = +inf, O = 0).
The kernel's running maximum mi cancels in exact arithmetic, so the softmax is written here without it:
O = sum(2**s V) / (sum(2**s) + 2**(attn_sink*log2(e))).
Scores use the stored sm_scale_div_log2 scalar with exp2, exactly as the kernel does; max_logits uses the f32 value
of CUDART_LN2_F and the sink uses the f32 value of CUDART_L2E_F, as the kernel does.
Q, KV and indices are strided views (tests/kernelkit non_contiguousify): the reference receives their contiguous
storage tensors and slices the logical blocks with the kernel's h_q and topk parameters (h_kv = 1).
"""
import torch

LN2 = 0.6931471824645996   # CUDART_LN2_F (math_constants.h: 6.931471805e-1F) as float32
L2E = 1.4426950216293335   # CUDART_L2E_F (math_constants.h: 1.442695041F) as float32


def reference(Q, KV, indices, attn_sink, topk_length, sm_scale_log2, h_q, topk):
    # Q: bf16 storage [s_q, h_q+1, 640], logical block [s_q, h_q, 512].  KV: bf16 storage [s_kv, 2, 640], logical
    # block [s_kv, h_kv=1, 512].  indices: int32 storage [s_q, 2, topk+128], logical block [s_q, h_kv=1, topk].
    # attn_sink: [h_q] float32 (natural-log units, may be +-inf).  topk_length: [s_q] int32.
    # sm_scale_log2 = sm_scale * log2(e).
    Q = Q[:, :h_q, :512].contiguous()
    KV = KV[:, :1, :512].contiguous()
    indices = indices[:, :1, :topk].contiguous()
    s_q, h_q, d_qk = Q.shape
    s_kv = KV.shape[0]
    idx = indices.reshape(s_q, topk)
    position = torch.arange(topk, dtype=torch.int32, device=idx.device).reshape(1, topk)
    valid = (idx >= 0) & (idx < s_kv) & (position < topk_length.reshape(s_q, 1))   # [s_q, topk]
    rows = torch.where(valid, idx, 0).reshape(-1).long()
    gathered = KV.reshape(s_kv, d_qk).index_select(0, rows).reshape(s_q, topk, d_qk).float()
    score = (Q.float() @ gathered.transpose(-1, -2)) * sm_scale_log2         # [s_q, h_q, topk], log2 units
    masked = torch.where(valid.unsqueeze(1), score, float("-inf"))
    max_logits = masked.amax(dim=-1) * LN2                                   # [s_q, h_q]
    weight = torch.where(valid.unsqueeze(1), torch.exp2(score), 0.0)
    total = weight.sum(dim=-1)                                               # [s_q, h_q]
    sink = torch.exp2(attn_sink * L2E).reshape(1, h_q)                       # e**attn_sink: -inf -> 0, +inf -> inf
    scale = torch.where(total == 0.0, 0.0, torch.reciprocal(total + sink))  # V = the whole 512-dim token
    out = (weight @ gathered) * scale.unsqueeze(-1)
    lse = torch.where(total == 0.0, float("inf"), torch.log(total))
    return out.to(torch.bfloat16), max_logits, lse
