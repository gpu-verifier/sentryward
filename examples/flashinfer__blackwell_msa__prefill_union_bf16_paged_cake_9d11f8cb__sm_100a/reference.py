# Source: https://github.com/flashinfer-ai/flashinfer/blob/47e64b0e96c446eaba90e40a1ecdf5891aaef92d/csrc/blackwell_msa/cake_blackwell_msa_9d11f8cbc9df656a767b_kernel.cu (softmax role L507-L1265, correction role L1266-L1490: out/lse L1420-L1486, load_warp page lookup L2046-L2189)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/47e64b0e96c446eaba90e40a1ecdf5891aaef92d/flashinfer/msa_ops/_blackwell_sm100.py (_prepare_layout L939-L1001, _prefill_route L1008-L1029, _run_prefill L2755-L2865, blackwell_msa_sparse_attention L3369-L3619)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/47e64b0e96c446eaba90e40a1ecdf5891aaef92d/tests/msa_ops/test_blackwell_msa_sm100.py (case prefill-paged-bf16-ragged L449-L464, _make_q2k L697-L731, _make_paged_cache L734-L768, _reference_attention L894-L1040)
"""Block-sparse (q2k top-k KV blocks) union prefill attention, paged KV layout.

Same attention as the flat route; logical KV block n of request b is the
physical page page_table[b, n] of k/v [num_pages, num_kv_heads, 128, D].
Kernel page handling (load_warp): a negative page id is replaced by 0; the TMA
coordinate page * num_kv_heads + kv_head at or beyond the descriptor extent
num_pages * num_kv_heads reads zeros (oob fill zero). Both are expressed with
torch.where. Requests are delimited by cu_seqlens_q and kv_lens (fixed shape
arrays; values enter as scalars). Causal masking uses the derived q offset
kv_len - q_len: query row i of the request sees tokens t <= kv_len - q_len + i.
A query tile only visits blocks below ceil((offset + q_len) / 128), which
contains every visible token. weight = exp2(softmax_scale_log2 * (q . k));
out = weight @ v / sum(weight), zero for a row without a selected key;
lse = ln(sum(weight)), -inf for such a row.
"""
import torch

BLOCK = 128


def _request(q, k, v, q2k_indices, page_table, softmax_scale_log2, causal, b, q_begin, q_end, kv_len):
    num_pages, num_kv_heads, page_size, head_dim = k.shape
    num_q_heads = q.shape[1]
    group = num_q_heads // num_kv_heads
    topk = q2k_indices.shape[2]
    q_len = q_end - q_begin
    n_blocks = (kv_len + BLOCK - 1) // BLOCK
    logical = page_table[b, :n_blocks]
    physical = torch.where(logical < 0, 0, logical)
    resident = (physical < num_pages).reshape(n_blocks, 1, 1)
    page = physical.clamp(0, num_pages - 1).long()
    block_rows = torch.arange(n_blocks, dtype=torch.int32, device=q.device).reshape(n_blocks, 1)
    token_block = (torch.zeros((n_blocks, BLOCK), dtype=torch.int32, device=q.device)
                   + block_rows).reshape(1, n_blocks * BLOCK)[:, :kv_len]
    token = torch.arange(kv_len, dtype=torch.int32, device=q.device).reshape(1, kv_len)
    position = torch.arange(q_len, dtype=torch.int32, device=q.device).reshape(q_len, 1) + (kv_len - q_len)

    def gather(cache, kv_head):
        selected = cache.index_select(0, page)[:, kv_head].float()
        selected = torch.where(resident, selected, 0.0)
        return selected.reshape(n_blocks * page_size, head_dim)[:kv_len]

    outs, lses = [], []
    for head in range(num_q_heads):
        kv_head = head // group
        query = q[q_begin:q_end, head, :].float()
        key = gather(k, kv_head)
        value = gather(v, kv_head)
        selection = q2k_indices[kv_head, q_begin:q_end, :]
        allowed = selection[:, 0:1] == token_block
        for slot in range(1, topk):
            allowed = allowed | (selection[:, slot:slot + 1] == token_block)
        if causal:
            allowed = allowed & (token <= position)
        weight = torch.where(allowed, torch.exp2((query @ key.transpose(0, 1)) * softmax_scale_log2), 0.0)
        total = weight.sum(dim=1, keepdim=True)
        nonempty = total > 0
        inv_total = torch.where(nonempty, 1.0 / torch.where(nonempty, total, 1.0), 0.0)
        outs.append((weight @ value) * inv_total)
        lses.append(torch.where(nonempty, torch.log(total), -float("inf")).reshape(q_len))
    return torch.stack(outs, dim=1), torch.stack(lses, dim=1)


def reference(q, k, v, q2k_indices, page_table, softmax_scale_log2, causal,
              cu_q_0, cu_q_1, cu_q_2, kv_len_0, kv_len_1):
    cu_q = (cu_q_0, cu_q_1, cu_q_2)
    kv_lens = (kv_len_0, kv_len_1)
    outs, lses = [], []
    for b in range(2):
        out, lse = _request(q, k, v, q2k_indices, page_table, softmax_scale_log2, causal,
                            b, cu_q[b], cu_q[b + 1], kv_lens[b])
        outs.append(out)
        lses.append(lse)
    return torch.cat(outs, dim=0).to(q.dtype), torch.cat(lses, dim=0)
