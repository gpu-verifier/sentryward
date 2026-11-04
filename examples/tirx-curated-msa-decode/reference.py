"""TIRx MiniMax sparse-attention decode, flat bf16 (curated_msa_decode_multishape, decode_bf16_b128_q1_kv4096_h64).

Source: https://github.com/mlc-ai/TIRx-kernels/blob/ff128be7470d674a94169e12c1556590e4046cba/tirx_kernels/msa/msa_decode_multishape.py
(config BENCH_CONFIGS L7989; inputs prepare_data L8124-L8203 and _make_q2k_indices L8092-L8104; oracle
_reference_output L8223-L8264). The recorded revision 532949266f is not public; ff128be7 is the earliest public one.

Upstream oracle: per request b, query row s and kv head h (query heads h * group .. h * group + group - 1), the key
tokens are the request-local positions p in [0, kv_len) whose 128-token block p // 128 is in the selected set
{q2k_indices[h, row, j]} and that satisfy p <= kv_len - seqlen_q + s (bottom-right causal boundary). The output is the
softmax-weighted mean of v over those tokens, and 0 for a row without one; it is cast to the query dtype.
kv_len = cu_seqlens_k[b + 1] - cu_seqlens_k[b]; the token lives at row cu_seqlens_k[b] + p of k and v.

Form: the selected blocks are gathered slot by slot (a block id below 0 selects nothing). A slot whose block id
already occurs in an earlier slot of the same row is dropped, so every block of the selected set counts once, as in
the oracle's `any` over slots. softmax(scale * q.k) is written as exp2(scale_log2 * q.k) normalized, with
scale_log2 = softmax_scale * log2(e) (the kernel argument p12); the running-max offset is abstracted (ideal real).
"""
import torch

BLOCK = 128


def reference(Q, K, V, cu_seqlens_k, q2k_indices, scale_log2):
    total_q, num_q_heads, head_dim = Q.shape
    total_k, num_kv_heads = K.shape[0], K.shape[1]
    group = num_q_heads // num_kv_heads
    topk = q2k_indices.shape[2]
    batch = cu_seqlens_k.shape[0] - 1
    seqlen_q = total_q // batch
    slots = topk * BLOCK
    device = Q.device

    start = cu_seqlens_k[:batch].long().reshape(batch, 1, 1, 1, 1)
    length = (cu_seqlens_k[1:] - cu_seqlens_k[:batch]).long().reshape(batch, 1, 1, 1, 1)
    # [b, s, h, j]: the block id of selection slot j of query row b * seqlen_q + s and kv head h.
    selected = q2k_indices.long().reshape(num_kv_heads, batch, seqlen_q, topk).permute(1, 2, 0, 3)
    slot = torch.arange(topk, device=device)
    earlier = slot.reshape(1, topk) < slot.reshape(topk, 1)                 # [j, i]: i < j
    repeated = (selected.unsqueeze(-1) == selected.unsqueeze(-2)) & earlier
    first = torch.where(repeated, 1, 0).sum(dim=-1) == 0                    # [b, s, h, j]
    sel = selected.unsqueeze(-1)                                             # [b, s, h, j, 1]
    t = torch.arange(BLOCK, device=device)
    position = sel * BLOCK + t                                               # request-local token position
    query = torch.arange(seqlen_q, device=device).reshape(1, seqlen_q, 1, 1, 1)
    valid = (sel >= 0) & first.unsqueeze(-1) & (position <= length - seqlen_q + query)
    token = torch.where(valid, start + position, 0)
    head = torch.arange(num_kv_heads, device=device).reshape(1, 1, num_kv_heads, 1, 1)
    row = token * num_kv_heads + head                                        # row of K.reshape(-1, head_dim)

    k = K.reshape(total_k * num_kv_heads, head_dim)[row].float().reshape(batch, seqlen_q, num_kv_heads, slots, head_dim)
    v = V.reshape(total_k * num_kv_heads, head_dim)[row].float().reshape(batch, seqlen_q, num_kv_heads, slots, head_dim)
    q = Q.float().reshape(batch, seqlen_q, num_kv_heads, group, head_dim)

    score = q @ k.transpose(-1, -2)                                          # [b, s, h, g, slots]
    mask = valid.reshape(batch, seqlen_q, num_kv_heads, 1, slots)
    weight = torch.where(mask, torch.exp2(score * scale_log2), 0.0)
    total = weight.sum(dim=-1, keepdim=True)
    nonempty = total > 0
    inv_total = torch.where(nonempty, 1.0 / torch.where(nonempty, total, 1.0), 0.0)
    out = (weight @ v) * inv_total
    return out.reshape(total_q, num_q_heads, head_dim).to(Q.dtype)
