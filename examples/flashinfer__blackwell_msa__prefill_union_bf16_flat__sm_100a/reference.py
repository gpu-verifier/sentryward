# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/blackwell_msa/sm100a/blackwell_msa_prefill_union_bf16_flat.cu
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/blackwell_msa/sm100a/blackwell_msa_prefill_union_bf16_flat_binding.cu
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/tests/msa_ops/test_blackwell_msa_sm100.py
"""Block-sparse (q2k top-k KV blocks) prefill attention, flat KV layout.

Query token t (absolute row q_begin + t) of q head h attends to key token j of
kv head h // group iff the 128-token KV block j // 128 appears in any of the
top-k slots q2k_indices[kv_head, q_begin + t, :]. Invalid (-1 or other
non-matching) slots select nothing. A row with no selected key outputs zero
(kernel: inv_sum = 0 when the row sum is not positive). Scores use the
kernel's base-2 scale: weight = exp2(softmax_scale_log2 * (q . k)).
The row maximum is omitted; it cancels under ideal-real semantics.
"""
import torch

BLOCK = 128


def reference(q, k, v, q2k_indices, cu_seqlens_q, cu_seqlens_k, kv_lens,
              softmax_scale_log2, q_begin, q_end, k_start, kv_len, launched_heads):
    # cu_seqlens_q / cu_seqlens_k / kv_lens are the input's fixed shape arrays;
    # their concrete values enter through q_begin, q_end, k_start and kv_len.
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group = num_q_heads // num_kv_heads
    topk = q2k_indices.shape[2]
    n_blocks = (kv_len + BLOCK - 1) // BLOCK
    block_rows = torch.arange(n_blocks, dtype=torch.int32, device=q.device).reshape(n_blocks, 1)
    token_block = (torch.zeros((n_blocks, BLOCK), dtype=torch.int32, device=q.device)
                   + block_rows).reshape(1, n_blocks * BLOCK)[:, :kv_len]
    heads = []
    for head in range(launched_heads):
        kv_head = head // group
        query = q[q_begin:q_end, head, :].float()
        key = k[k_start:k_start + kv_len, kv_head, :].float()
        value = v[k_start:k_start + kv_len, kv_head, :].float()
        selection = q2k_indices[kv_head, q_begin:q_end, :]
        allowed = selection[:, 0:1] == token_block
        for slot in range(1, topk):
            allowed = allowed | (selection[:, slot:slot + 1] == token_block)
        weight = torch.where(allowed, torch.exp2((query @ key.T) * softmax_scale_log2), 0.0)
        total = weight.sum(dim=-1, keepdim=True)
        inv_total = torch.where(total > 0, 1.0 / torch.where(total > 0, total, 1.0), 0.0)
        heads.append((weight @ value) * inv_total)
    return torch.stack(heads, dim=1).to(torch.bfloat16)
