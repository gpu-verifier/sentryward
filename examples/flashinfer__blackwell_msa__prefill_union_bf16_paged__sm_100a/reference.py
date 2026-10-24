# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/blackwell_msa/sm100a/blackwell_msa_prefill_union_bf16_paged.cu
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/blackwell_msa/sm100a/blackwell_msa_prefill_union_bf16_paged_binding.cu
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/tests/msa_ops/test_blackwell_msa_sm100.py
"""Block-sparse (q2k top-k KV blocks) prefill attention, paged KV layout.

Same attention as the flat route; logical KV block b of batch 0 is the
physical page page_table[0, b] of k/v [num_pages, num_kv_heads, 128, D].
Kernel sentinel handling (load_warp): a negative page id is replaced by 0;
the TMA coordinate page * num_kv_heads + kv_head beyond the descriptor extent
num_pages * num_kv_heads reads zeros (oob_fill zero). Both are expressed with
torch.where; the page id is never hardcoded.
"""
import torch

BLOCK = 128


def reference(q, k, v, q2k_indices, page_table, cu_seqlens_q, kv_lens,
              softmax_scale_log2, q_begin, q_end, kv_len, launched_heads):
    # cu_seqlens_q / kv_lens are the input's fixed shape arrays; their concrete
    # values enter through q_begin, q_end and kv_len.
    num_pages, num_kv_heads, page_size, head_dim = k.shape
    num_q_heads = q.shape[1]
    group = num_q_heads // num_kv_heads
    topk = q2k_indices.shape[2]
    n_blocks = (kv_len + BLOCK - 1) // BLOCK
    logical = page_table[0, :n_blocks]
    physical = torch.where(logical < 0, 0, logical)
    resident = (physical < num_pages).reshape(n_blocks, 1, 1)
    page = physical.clamp(0, num_pages - 1).long()
    block_rows = torch.arange(n_blocks, dtype=torch.int32, device=q.device).reshape(n_blocks, 1)
    token_block = (torch.zeros((n_blocks, BLOCK), dtype=torch.int32, device=q.device)
                   + block_rows).reshape(1, n_blocks * BLOCK)[:, :kv_len]

    def gather(cache, kv_head):
        selected = cache.index_select(0, page)[:, kv_head].float()
        selected = torch.where(resident, selected, 0.0)
        return selected.reshape(n_blocks * page_size, head_dim)[:kv_len]

    heads = []
    for head in range(launched_heads):
        kv_head = head // group
        query = q[q_begin:q_end, head, :].float()
        key = gather(k, kv_head)
        value = gather(v, kv_head)
        selection = q2k_indices[kv_head, q_begin:q_end, :]
        allowed = selection[:, 0:1] == token_block
        for slot in range(1, topk):
            allowed = allowed | (selection[:, slot:slot + 1] == token_block)
        weight = torch.where(allowed, torch.exp2((query @ key.T) * softmax_scale_log2), 0.0)
        total = weight.sum(dim=-1, keepdim=True)
        inv_total = torch.where(total > 0, 1.0 / torch.where(total > 0, total, 1.0), 0.0)
        heads.append((weight @ value) * inv_total)
    return torch.stack(heads, dim=1).to(torch.bfloat16)
