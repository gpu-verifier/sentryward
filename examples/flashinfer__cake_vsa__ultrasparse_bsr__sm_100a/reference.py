"""CAKE VSA ultrasparse BSR forward: six 128-token KV blocks per query block.

Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/cake_vsa/cake_vsa_ultrasparse_bsr_sm_100a.cu
Host ABI: csrc/cake_vsa/cake_vsa_ultrasparse_bsr_host.cpp; wrapper and contract:
flashinfer/cake_vsa.py and tests/attention/test_cake_vsa.py (same commit).

q/out are bf16 [M, H, 128] (out is a plain pointer), k/v are bf16 [N, H, 128]
and bsr_indices is int32 [M/128, 6], shared by all heads. Query block b
attends the KV blocks bsr_indices[b, j], j = 0..5. The kernel masks the whole
block to -inf when nb*128 - block*128 <= 0 (block >= nb). A key token outside
[0, N) is read through the TMA zero fill (K = V = 0). Every list entry is
processed on its own, so a repeated block id is counted once per entry. A
row with no live entry gets sum 0 and the kernel writes 0. The kernel uses
kv_head = q_head. LSE outputs are written only when their flags are nonzero;
this scenario passes 0 for both.
"""
import torch

BLOCK = 128
SELECTED = 6


def reference(q, k, v, bsr_indices, nb, scale_log2):
    seq_q, heads, dim = q.shape
    seq_kv = k.shape[0]
    q_blocks = bsr_indices.shape[0]
    device = q.device
    # [H, MB, 128, D]: query block b holds query rows b*128 .. b*128+127.
    query = q.float().reshape(q_blocks, BLOCK, heads, dim).permute(2, 0, 1, 3)
    block = bsr_indices.to(torch.int64).reshape(1, q_blocks, SELECTED, 1)
    offsets = torch.arange(BLOCK, device=device).reshape(1, 1, 1, BLOCK)
    token = block * BLOCK + offsets
    inside = (token >= 0) & (token < seq_kv)
    safe = torch.where(inside, token, 0)
    head = torch.arange(heads, device=device).reshape(heads, 1, 1, 1)
    keep = inside.unsqueeze(-1)
    key = torch.where(keep, k[safe, head].float(), 0.0)
    value = torch.where(keep, v[safe, head].float(), 0.0)
    key = key.reshape(heads, q_blocks, SELECTED * BLOCK, dim)
    value = value.reshape(heads, q_blocks, SELECTED * BLOCK, dim)
    # valid_cols = clamp(nb*128 - block*128, 0, 128): live iff block < nb.
    live_block = block < nb
    live = (live_block & (offsets < BLOCK)).reshape(1, q_blocks, 1, SELECTED * BLOCK)
    # Ideal-real softmax: the running-max offset and the bf16 P rounding of
    # the kernel are abstracted, as in examples/_reference/pytorch_models.py.
    score = (query @ key.transpose(-1, -2)) * scale_log2
    weight = torch.where(live, torch.exp2(score), 0.0)
    live_count = torch.where(live_block, 1, 0).reshape(1, q_blocks, SELECTED).sum(dim=-1)
    any_live = (live_count > 0).reshape(1, q_blocks, 1, 1)
    total = torch.where(any_live, weight.sum(dim=-1, keepdim=True), 1.0)
    out = (weight @ value) / total
    # [MB, 128, H, D] equals the row-major [M, H, D] output.
    return out.permute(1, 2, 0, 3).to(q.dtype)
