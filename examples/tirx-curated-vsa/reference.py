"""TIRx VSA blk128 forward (curated_vsa_multishape, pooled_blk128_s16384_topk64).

Source: https://github.com/mlc-ai/TIRx-kernels/blob/ff128be7470d674a94169e12c1556590e4046cba/tirx_kernels/vsa/vsa_multishape.py
(make_attention_kernel_blk128 L313-1356; oracle _reference_output L2912-2942).
The device code is unchanged at 01130e3bb1cf8a009fc1e7e6db2a860cd51df9ed.

q/k/v/out are bf16 [S, H, 128]; q2k_indices is int32 [H, S/128, topk]. Task
(head, tile) takes query rows tile*128 .. tile*128+127 of that head and attends
the 128-token KV blocks q2k_indices[head, tile, j] for every j < topk (topk is
even, complete_pairs, so no stage is masked). Every list entry is processed on
its own, as in the upstream oracle, so a repeated id counts twice. A key token
outside [0, S) is read through the TMA zero fill (K = V = 0). With the full
148-CTA grid all S/128 * H tiles are written.
"""
import torch

BLOCK = 128


def reference(q, k, v, q2k_indices, scale_log2):
    seq, heads, dim = q.shape
    seq_kv = k.shape[0]
    _, tiles, topk = q2k_indices.shape
    device = q.device
    # [H, T, 128, D]: tile t holds query rows t*128 .. t*128+127.
    query = q.float().reshape(tiles, BLOCK, heads, dim).permute(2, 0, 1, 3)
    offsets = torch.arange(BLOCK, device=device).reshape(1, 1, 1, BLOCK)
    token = q2k_indices.to(torch.int64).reshape(heads, tiles, topk, 1) * BLOCK + offsets
    inside = (token >= 0) & (token < seq_kv)
    safe = torch.where(inside, token, 0)
    head = torch.arange(heads, device=device).reshape(heads, 1, 1, 1)
    keep = inside.unsqueeze(-1)
    key = torch.where(keep, k[safe, head].float(), 0.0)
    value = torch.where(keep, v[safe, head].float(), 0.0)
    key = key.reshape(heads, tiles, topk * BLOCK, dim)
    value = value.reshape(heads, tiles, topk * BLOCK, dim)
    # Ideal-real softmax: the running-max offset and the bf16 P rounding of
    # the kernel are abstracted, as in examples/_reference/pytorch_models.py.
    weight = torch.exp2((query @ key.transpose(-1, -2)) * scale_log2)
    out = (weight @ value) / weight.sum(dim=-1, keepdim=True)
    # [T, 128, H, D] equals the row-major [S, H, D] output.
    return out.permute(1, 2, 0, 3).to(q.dtype)
