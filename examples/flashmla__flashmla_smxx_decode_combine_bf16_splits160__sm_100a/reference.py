"""FlashMLA flash_fwd_mla_combine_kernel<bf16, 512, 8, MAX_SPLITS=160, 256, false> (no attention sink).

Source (FlashMLA 395cb3afbb331fef7b1b21e9340861b9436b42d8):
  csrc/cuda_kernels/smxx/decode/combine/combine.cu:64-69 (splits [num_splits[b], num_splits[b+1]); a request with
    one split returns early and keeps the decode kernel's O/LSE), :100-129 (max over the split LSEs, -inf -> 0;
    sum of exp2(lse - max); global = +inf if the sum is 0 else log2(sum) + max; LSE = global / M_LOG2E),
    :131-142 (attention sink: attn_sink is null in this scenario), :143-147 (scale = exp2(lse - global)),
    :152-184 (O = sum_split scale * o_accum, rounded to bf16);
  csrc/api/sparse_decode.cpp:379-436 (lse_accum/o_accum [b + num_sm_parts, s_q, h_q(, d_v)], combine launch).
lse_accum is in log2 units (the split-KV decode kernel's output); LSE is written in natural-log units.
log2f(sum) (combine.cu:127) is written as log(sum) / ln(2) with a symbolic ln(2) (log of the constant 2),
because the capture does not support aten.log2; a rounded ln 2 literal would denote a different real function.
"""
import torch

MAX_SPLITS = 160
M_LOG2E_F32 = 1.4426950216293335   # (float)M_LOG2E


def reference(lse_accum, o_accum, num_splits, out_in, lse_in):
    b = num_splits.shape[0] - 1
    start = num_splits[:-1].long()
    count = (num_splits[1:] - num_splits[:-1]).long()                         # [b]
    s = torch.arange(MAX_SPLITS, device=num_splits.device)
    valid = s.reshape(1, -1) < count.reshape(-1, 1)                           # [b, MAX_SPLITS]
    idx = torch.where(valid, start.reshape(-1, 1) + s.reshape(1, -1), torch.zeros_like(valid, dtype=torch.long))
    local = lse_accum.index_select(0, idx.reshape(-1)).reshape(b, MAX_SPLITS, *lse_accum.shape[1:])
    local = torch.where(valid.reshape(b, MAX_SPLITS, 1, 1), local, float("-inf"))   # [b, S, s_q, h_q]
    mx = local.amax(dim=1)
    mx = torch.where(mx == float("-inf"), 0.0, mx)
    total = torch.exp2(local - mx.unsqueeze(1)).sum(dim=1)
    ln2 = torch.log(torch.ones_like(total) * 2.0)
    glob = torch.where(total == 0.0, float("inf"), torch.log(total) / ln2 + mx)   # [b, s_q, h_q], log2 units
    scale = torch.exp2(local - glob.unsqueeze(1))                              # [b, S, s_q, h_q]
    parts = o_accum.index_select(0, idx.reshape(-1)).reshape(b, MAX_SPLITS, *o_accum.shape[1:])
    parts = torch.where(valid.reshape(b, MAX_SPLITS, 1, 1, 1), parts, 0.0)
    combined = (scale.unsqueeze(-1) * parts).sum(dim=1)                        # [b, s_q, h_q, d_v]
    keep = (count == 1).reshape(b, 1, 1)
    out = torch.where(keep.unsqueeze(-1), out_in.float(), combined).to(torch.bfloat16)
    lse = torch.where(keep, lse_in, glob / M_LOG2E_F32)
    return out, lse
