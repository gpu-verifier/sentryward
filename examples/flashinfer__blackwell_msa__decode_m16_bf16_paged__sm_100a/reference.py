# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/blackwell_msa/sm100a/blackwell_msa_decode_m16_bf16_paged.cu (kernel_minimax_sparse_decode_m16_paged_sm100, L409-1972; producer L1636-1953)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/blackwell_msa/sm100a/blackwell_msa_decode_m16_bf16_paged_binding.cu (TMA maps L136-320, Run L397-663)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/flashinfer/msa_ops/_blackwell_sm100.py (_prepare_layout L469-535, _run_decode_module L1850-1985, blackwell_msa_sparse_decode_attention L2323-2636)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/tests/msa_ops/test_blackwell_msa_sm100.py (_make_paged_cache L637-671, _reference_attention L797-906)
#
# Direct M16 sparse decode, paged K/V (num_pages, num_kv_heads, 128, 128).  For query row r
# (batch = r // seqlen_q), KV head h and selection slot j: sel = q2k_indices[h, r, j] (task_kind
# slot), physical page = page_table.flat[batch * max_pages + sel] (kv_indices slot), token t of
# that page is valid iff sel >= 0, page >= 0 and sel*128 + t < kv_lens[batch] (task_kv_head slot).
# O = softmax-weighted V over all valid (j, t) slots; msa_lse = ln(sum exp2(softmax_scale_log2 *
# q.k)); empty rows give O = 0 and LSE = -inf.  Slots are summed per selection position, as the
# kernel does; with distinct selections this equals the upstream union-of-blocks reference.
import torch


def reference(Q, K, V, page_table, q2k_indices, kv_lens, softmax_scale_log2, seqlen_q, max_pages, is_causal):
    if is_causal:
        raise NotImplementedError("causal masking is outside the recorded scenario (is_causal = 0)")
    total_q, num_q_heads, head_dim = Q.shape
    num_pages, num_kv_heads, page_size = K.shape[0], K.shape[1], K.shape[2]
    group = num_q_heads // num_kv_heads
    topk = q2k_indices.shape[2]
    batch = total_q // seqlen_q
    slots = topk * page_size

    # [b, s, h, j] selection ids
    sel = q2k_indices.long().reshape(num_kv_heads, batch, seqlen_q, topk).permute(1, 2, 0, 3)
    selected = sel >= 0
    batch_index = torch.arange(batch, device=Q.device).reshape(batch, 1, 1, 1)
    table_index = batch_index * max_pages + torch.where(selected, sel, 0)
    raw_page = page_table.reshape(-1).long()[table_index]
    page_ok = selected & (raw_page >= 0)
    page = torch.where(page_ok, raw_page, 0).unsqueeze(-1)       # [b, s, h, j, 1]

    t = torch.arange(page_size, device=Q.device)
    length = kv_lens.long().reshape(batch, 1, 1, 1, 1)
    valid = page_ok.unsqueeze(-1) & (sel.unsqueeze(-1) * page_size + t < length)   # [b, s, h, j, t]
    head = torch.arange(num_kv_heads, device=Q.device).reshape(1, 1, num_kv_heads, 1, 1)
    row = (page * num_kv_heads + head) * page_size + t          # row of K.reshape(-1, head_dim)

    rows = num_pages * num_kv_heads * page_size
    k = K.reshape(rows, head_dim)[row].float().reshape(batch, seqlen_q, num_kv_heads, slots, head_dim)
    v = V.reshape(rows, head_dim)[row].float().reshape(batch, seqlen_q, num_kv_heads, slots, head_dim)
    q = Q.float().reshape(batch, seqlen_q, num_kv_heads, group, head_dim)

    score = q @ k.transpose(-1, -2)                             # [b, s, h, g, slots]
    mask = valid.reshape(batch, seqlen_q, num_kv_heads, 1, slots)
    weight = torch.where(mask, torch.exp2(score * softmax_scale_log2), 0.0)
    total = weight.sum(dim=-1, keepdim=True)
    nonempty = total > 0
    out = torch.where(nonempty, (weight @ v) / total, 0.0)
    lse = torch.where(nonempty, torch.log(total), -float("inf"))
    return (out.reshape(total_q, num_q_heads, head_dim).to(Q.dtype),
            lse.reshape(total_q, num_q_heads))
