# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/blackwell_msa/sm100a/blackwell_msa_decode_m16_bf16_flat.cu (kernel_minimax_sparse_decode_m16_flat_sm100, L391-1792)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/blackwell_msa/sm100a/blackwell_msa_decode_m16_bf16_flat_binding.cu (TMA maps L136-320, Run L397-657)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/flashinfer/msa_ops/_blackwell_sm100.py (_prepare_layout L469-535, _run_decode_module L1850-1985, blackwell_msa_sparse_decode_attention L2323-2636)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/tests/msa_ops/test_blackwell_msa_sm100.py (_reference_attention L797-906)
#
# Direct M16 sparse decode, flat K/V (total_k, num_kv_heads, 128).  For query row r
# (batch = r // seqlen_q), KV head h and selection slot j, the kernel reads block
# sel = q2k_indices[h, r, j] (q2k is passed in the task_kind slot) and the 128 tokens
# kv_indptr[batch] + sel*128 + t; token t is valid iff sel >= 0 and sel*128 + t < kv_len,
# kv_len = kv_indptr[batch+1] - kv_indptr[batch].  O = softmax-weighted V over all valid
# (j, t) slots; msa_lse = ln(sum exp2(softmax_scale_log2 * q.k)); empty rows give O = 0 and
# LSE = -inf.  Slots are summed per selection position, exactly as the kernel does; with
# distinct selections this equals the upstream union-of-blocks reference.
import torch


def reference(Q, K, V, kv_indptr, q2k_indices, softmax_scale_log2, seqlen_q, is_causal):
    if is_causal:
        raise NotImplementedError("causal masking is outside the recorded scenario (is_causal = 0)")
    block = 128
    total_q, num_q_heads, head_dim = Q.shape
    total_k, num_kv_heads = K.shape[0], K.shape[1]
    group = num_q_heads // num_kv_heads
    topk = q2k_indices.shape[2]
    batch = total_q // seqlen_q
    slots = topk * block

    start = kv_indptr[:batch].long().reshape(batch, 1, 1, 1, 1)
    length = (kv_indptr[1:batch + 1] - kv_indptr[:batch]).long().reshape(batch, 1, 1, 1, 1)
    # [b, s, h, j] selection ids
    sel = q2k_indices.long().reshape(num_kv_heads, batch, seqlen_q, topk).permute(1, 2, 0, 3).unsqueeze(-1)
    t = torch.arange(block, device=Q.device)
    logical = sel * block + t                                   # token position inside the request
    valid = (sel >= 0) & (logical < length)                     # [b, s, h, j, t]
    token = torch.where(valid, start + logical, 0)
    head = torch.arange(num_kv_heads, device=Q.device).reshape(1, 1, num_kv_heads, 1, 1)
    row = token * num_kv_heads + head                           # row of K.reshape(-1, head_dim)

    k = K.reshape(total_k * num_kv_heads, head_dim)[row].float().reshape(batch, seqlen_q, num_kv_heads, slots, head_dim)
    v = V.reshape(total_k * num_kv_heads, head_dim)[row].float().reshape(batch, seqlen_q, num_kv_heads, slots, head_dim)
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
