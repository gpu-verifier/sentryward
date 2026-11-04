# Source: https://github.com/flashinfer-ai/flashinfer/blob/47e64b0e96c446eaba90e40a1ecdf5891aaef92d/csrc/blackwell_msa/cake_blackwell_msa_1a404b555f04cdde1d2c_kernel.cu (softmax role L507-L1265, correction role L1266-L1490: out/lse L1420-L1486, load_warp L2046-L2179)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/47e64b0e96c446eaba90e40a1ecdf5891aaef92d/flashinfer/msa_ops/_blackwell_sm100.py (_prepare_layout L939-L1001, _prefill_route L1008-L1029, _run_prefill L2755-L2865, blackwell_msa_sparse_attention L3369-L3619)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/47e64b0e96c446eaba90e40a1ecdf5891aaef92d/tests/msa_ops/test_blackwell_msa_sm100.py (case prefill-flat-bf16-ragged-noncausal L127-L142, _make_q2k L697-L731, _reference_attention L894-L1040)
"""Block-sparse (q2k top-k KV blocks) union prefill attention, flat KV layout.

Requests are delimited by cu_seqlens_q / cu_seqlens_k (fixed shape arrays of the
input; their values enter as the scalars cu_q_* / cu_k_*). Query row r of
request b and q head h attend to key token t of KV head h // group iff the
128-token block t // 128 appears in some slot of q2k_indices[h // group, r, :]
and t < kv_len (non-causal; with causal set, also t <= kv_len - q_len + local
row, the derived q offset). Slots holding -1 or a block outside the request
select nothing. Scores use the kernel's base-2 scale:
weight = exp2(softmax_scale_log2 * (q . k)). out = weight @ v / sum(weight),
zero for a row without a selected key (kernel inv_sum = 0); lse =
ln(sum(weight)), -inf for such a row. The row maximum cancels under
ideal-real semantics.
"""
import torch

BLOCK = 128


def _request(q, k, v, q2k_indices, softmax_scale_log2, causal, q_begin, q_end, k_start, kv_len):
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group = num_q_heads // num_kv_heads
    topk = q2k_indices.shape[2]
    q_len = q_end - q_begin
    n_blocks = (kv_len + BLOCK - 1) // BLOCK
    block_rows = torch.arange(n_blocks, dtype=torch.int32, device=q.device).reshape(n_blocks, 1)
    token_block = (torch.zeros((n_blocks, BLOCK), dtype=torch.int32, device=q.device)
                   + block_rows).reshape(1, n_blocks * BLOCK)[:, :kv_len]
    token = torch.arange(kv_len, dtype=torch.int32, device=q.device).reshape(1, kv_len)
    position = torch.arange(q_len, dtype=torch.int32, device=q.device).reshape(q_len, 1) + (kv_len - q_len)
    outs, lses = [], []
    for head in range(num_q_heads):
        kv_head = head // group
        query = q[q_begin:q_end, head, :].float()
        key = k[k_start:k_start + kv_len, kv_head, :].float()
        value = v[k_start:k_start + kv_len, kv_head, :].float()
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


def reference(q, k, v, q2k_indices, softmax_scale_log2, causal,
              cu_q_0, cu_q_1, cu_q_2, cu_k_0, cu_k_1, cu_k_2):
    cu_q = (cu_q_0, cu_q_1, cu_q_2)
    cu_k = (cu_k_0, cu_k_1, cu_k_2)
    outs, lses = [], []
    for b in range(2):
        out, lse = _request(q, k, v, q2k_indices, softmax_scale_log2, causal,
                            cu_q[b], cu_q[b + 1], cu_k[b], cu_k[b + 1] - cu_k[b])
        outs.append(out)
        lses.append(lse)
    return torch.cat(outs, dim=0).to(q.dtype), torch.cat(lses, dim=0)
