# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/flashinfer/msa_ops/_blackwell_sm100_reverse_plan.py (build_reverse_schedule_inputs L212-L279: CSR rows row_linear = kv_block * batch_size + batch, payload local_q | slot << 24; build_bf16_paged_topk4_plan L391-L450: four live slots per query row and kv head)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/flashinfer/msa_ops/_blackwell_sm100.py (_exact_topk4_launch_parts L1380-L1513: partial_o fp8 [4, total_q, H, 128], partial_scale / partial_lse f32 [4, total_q, H])
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/blackwell_msa/sm100a/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_const4_reduce.cu (combine L49-L130: out = sum_s softmax_s(partial_lse) * partial_scale_s * fp8(partial_o_s))
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/blackwell_msa/sm100a/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4.cu (producer: edge decode L730-L760, partial row and per-row FP8 encoding L1018-L1112, page lookup L2281-L2289)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/tests/msa_ops/test_blackwell_msa_sm100.py (oracle _reference_attention L797-L906)
"""Reverse (KV-major) sparse prefill producer, paged bf16, top-k 4, GQA 4: the split-K partials.

The upstream oracle is block-sparse causal attention (`_reference_attention`). This kernel is the producer half of
the reverse route; the const4 reducer combines its partials into the oracle's output. Split s of query row q_abs and
query head g = h * 4 + j (kv head h) is the attention of that query over its s-th selected 128-token KV block alone:
  w_t = exp2(softmax_scale_log2 * q . k_t) over the visible tokens t of the block (position < kv_len and
  position <= kv_len - q_len + q_idx: bottom-right causal), O = sum_t w_t v_t / sum_t w_t (0 without a token),
  partial_lse = ln(sum_t w_t) (-inf without a token),
  partial_scale = max_d |O_d| / 448 and partial_o = e4m3(O * 448 / max_d |O_d|) (0 when O = 0),
so that the reducer's sum_s softmax_s(partial_lse) * partial_scale_s * partial_o_s is the oracle's output row.

The kernel does not receive q2k_indices but the planner's K-to-Q CSR: per kv head h, edges
row_ptr[h, r] <= e < row_ptr[h, r + 1] of row r = kv_block * batch_size + batch, each with the payload
q_idx | slot << 24 (q_abs = cu_seqlens_q[batch] + q_idx). The partial of (slot s, row q_abs, kv head h) is therefore
the edge of head h whose payload decodes to (s, q_abs); this is written as a sum over edges masked by that match.
Logical block n of a request reads physical page page_table[batch, n] (negative -> 0; beyond the pool -> zeros).
"""
import torch

BLOCK = 128
FP8_MAX = 448.0
QIDX_MASK = 0xFFFFFF
SLOT_SHIFT = 24
SLOT_MASK = 0xFF


def reference(q, k, v, k2q_row_ptr, k2q_qsplit_indices, page_table, cu_seqlens_q, kv_lens,
              softmax_scale_log2, topk):
    total_q, num_q_heads, head_dim = q.shape
    num_pages, num_kv_heads = k.shape[0], k.shape[1]
    group = num_q_heads // num_kv_heads
    batch_size = cu_seqlens_q.shape[0] - 1
    total_rows = k2q_row_ptr.shape[1] - 1
    nnz = k2q_qsplit_indices.shape[1]
    device = q.device

    # The CSR row of every edge, and that row's request and logical KV block.
    edge = torch.arange(nnz, device=device).reshape(1, nnz, 1)
    ends = k2q_row_ptr[:, 1:].long().reshape(num_kv_heads, 1, total_rows)
    row = torch.where(ends <= edge, 1, 0).sum(dim=-1)                        # [Hkv, E]
    kv_block = row // batch_size
    batch = row % batch_size
    packed = k2q_qsplit_indices.long()
    q_idx = packed & QIDX_MASK
    slot = (packed >> SLOT_SHIFT) & SLOT_MASK
    cu = cu_seqlens_q.long()
    q_abs = cu[batch] + q_idx
    q_len = (cu[1:] - cu[:batch_size])[batch]
    kv_len = kv_lens.long()[batch]

    head = torch.arange(num_kv_heads, device=device).reshape(num_kv_heads, 1)
    query = q.reshape(total_q, num_kv_heads, group, head_dim)[q_abs, head].float()   # [Hkv, E, G, D]
    logical = page_table.long()[batch, kv_block]
    physical = torch.where(logical < 0, 0, logical)
    resident = (physical < num_pages).reshape(num_kv_heads, nnz, 1, 1)
    page = physical.clamp(0, num_pages - 1)
    key = torch.where(resident, k[page, head].float(), 0.0)                  # [Hkv, E, BLOCK, D]
    value = torch.where(resident, v[page, head].float(), 0.0)
    position = kv_block.unsqueeze(-1) * BLOCK + torch.arange(BLOCK, device=device)
    visible = (position < kv_len.unsqueeze(-1)) & (position <= (kv_len - q_len + q_idx).unsqueeze(-1))

    score = query @ key.transpose(-1, -2)                                    # [Hkv, E, G, BLOCK]
    weight = torch.where(visible.unsqueeze(-2), torch.exp2(score * softmax_scale_log2), 0.0)
    total = weight.sum(dim=-1)                                               # [Hkv, E, G]
    nonempty = total > 0
    inv_total = torch.where(nonempty, 1.0 / torch.where(nonempty, total, 1.0), 0.0)
    out = (weight @ value) * inv_total.unsqueeze(-1)                         # [Hkv, E, G, D]
    lse = torch.where(nonempty, torch.log(torch.where(nonempty, total, 1.0)), -float("inf"))
    absmax = out.abs().amax(dim=-1)                                          # [Hkv, E, G]
    present = absmax > 0
    quant = torch.where(present, FP8_MAX / torch.where(present, absmax, 1.0), 0.0)
    encoded = (out * quant.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    scale = absmax / FP8_MAX

    # Placement: output (s, q_abs, h * G + j) takes the edge of head h whose payload decodes to (s, q_abs).
    split = torch.arange(topk, device=device).reshape(topk, 1, 1, 1)
    rows = torch.arange(total_q, device=device).reshape(1, total_q, 1, 1)
    match = (slot.reshape(1, 1, num_kv_heads, nnz) == split) & (q_abs.reshape(1, 1, num_kv_heads, nnz) == rows)

    def place(x):                                                            # [Hkv, E, G, ...] -> [S, T, Hq, ...]
        tail = x.shape[3:]
        hit = match.reshape(topk, total_q, num_kv_heads, nnz, *([1] * (1 + len(tail))))
        placed = torch.where(hit, x.unsqueeze(0).unsqueeze(0), 0.0).sum(dim=3)
        return placed.reshape(topk, total_q, num_q_heads, *tail)

    partial_o = place(encoded).to(torch.float8_e4m3fn)
    partial_scale = place(scale)
    partial_lse = place(lse)
    return partial_o, partial_scale, partial_lse
