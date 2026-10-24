"""CAKE VSA blk64 weight-stationary (M=64, N=256) forward: block-sparse attention.

Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/cake_vsa/cake_vsa_blk64_persistent_ws_m64n256_sm_100a.cu
Host ABI: csrc/cake_vsa/cake_vsa_blk64_persistent_ws_m64n256_host.cpp; wrapper and
contract: flashinfer/cake_vsa.py and tests/attention/test_cake_vsa.py (same commit).

q/out are bf16 [M, H, 128], k/v are bf16 [N, H, 128], q2k_indices is int32
[H, M/64, topk] and q2k_num is int32 [H, M/64]. Query block b of head h attends
the 64-token KV blocks q2k_indices[h, b, j] for j < 4 * (q2k_num[h, b] // 4):
the kernel consumes whole groups of four list entries. Every list entry is
processed on its own, so a repeated block id is counted once per entry. A key
token outside [0, N) is read through the TMA zero fill (K = V = 0). The kernel
never loads kv_block_lens, so it is not a reference input. LSE is written only
when return_lse != 0; this scenario passes 0.
"""
import torch

BLOCK = 64


def reference(q, k, v, q2k_indices, q2k_num, scale_log2):
    seq_q, heads, dim = q.shape
    seq_kv = k.shape[0]
    _, q_blocks, topk = q2k_indices.shape
    device = q.device
    # [H, MB, 64, D]: query block b holds query rows b*64 .. b*64+63.
    query = q.float().reshape(q_blocks, BLOCK, heads, dim).permute(2, 0, 1, 3)
    offsets = torch.arange(BLOCK, device=device).reshape(1, 1, 1, BLOCK)
    token = q2k_indices.to(torch.int64).reshape(heads, q_blocks, topk, 1) * BLOCK + offsets
    inside = (token >= 0) & (token < seq_kv)
    safe = torch.where(inside, token, 0)
    head = torch.arange(heads, device=device).reshape(heads, 1, 1, 1)
    keep = inside.unsqueeze(-1)
    key = torch.where(keep, k[safe, head].float(), 0.0)
    value = torch.where(keep, v[safe, head].float(), 0.0)
    key = key.reshape(heads, q_blocks, topk * BLOCK, dim)
    value = value.reshape(heads, q_blocks, topk * BLOCK, dim)
    # Only whole groups of four entries are consumed: 4 * (count / 4), which
    # equals count - count % 4 on the kernel's domain count >= 4.
    count = q2k_num.to(torch.int64).reshape(heads, q_blocks, 1, 1)
    slot = torch.arange(topk, device=device).reshape(1, 1, topk, 1)
    active = (slot < count - count % 4) & (offsets < BLOCK)
    active = active.reshape(heads, q_blocks, 1, topk * BLOCK)
    # Ideal-real softmax: the running-max offset and the bf16 P rounding of
    # the kernel are abstracted, as in examples/_reference/pytorch_models.py.
    score = (query @ key.transpose(-1, -2)) * scale_log2
    weight = torch.where(active, torch.exp2(score), 0.0)
    out = (weight @ value) / weight.sum(dim=-1, keepdim=True)
    # [MB, 64, H, D] equals the row-major [M, H, D] output.
    return out.permute(1, 2, 0, 3).to(q.dtype)
