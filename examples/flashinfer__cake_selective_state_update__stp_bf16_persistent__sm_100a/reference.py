"""Single-token bf16 selective state update (CAKE stp_bf16_persistent), state written to dst slot.

Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/cake_selective_state_update/cuda/cake_selective_state_update_stp_bf16_persistent.cu
  lines 272-565 (kernel_cake_selective_state_update_stp_bf16_persistent; work tiles
  bid, bid + gridDim.x, ... < total_head_tiles; per-tile math identical to stp_bf16_direct)
Host/dispatch: csrc/cake_selective_state_update/host/cake_selective_state_update_stp_bf16_persistent.cc
  lines 205-244 (4-D state tensor map), flashinfer/jit/mamba/cake_selective_state_update.py
  lines 334-411, 500-543 (x [batch, nheads, 128]; destination = source when dst is None).

Per batch b and head h (group g = h // (nheads // ngroups)), rows r and state columns s:
  dt' = dt[b, h] + dt_bias[h]; softplus: dt' <= 20 ? log(1 + exp(dt')) : dt'
  S'  = S[src_slot, h] * exp(A[h] * dt') + B[b, g, s] * dt' * x[b, h, r]
  y   = sum_s S'[r, s] * C[b, g, s] + D[h] * x[b, h, r]          (y uses the fp32 S')
  state[dst_slot, h] = bf16(S');  output[b, h, r] = bf16(y)
TMA semantics: slot coordinates are the int64 indices cast to int32; an out-of-range
source slot loads zeros, an out-of-range destination slot is not written.
State slots other than the destination keep their contents.
"""
import torch


def reference(state, x, dt, A, B, C, D, dt_bias, state_batch_indices,
              dst_state_batch_indices, nheads, ngroups, head_tiles, total_head_tiles,
              dt_softplus, has_z, disable_state_update):
    if has_z or disable_state_update:
        raise ValueError("binding covers has_z=0 and disable_state_update=0 only")
    slots, heads, dim, dstate = state.shape
    batch = x.shape[0]
    if heads != nheads or x.shape[1] != nheads or (dim, dstate) != (128, 128):
        raise ValueError("expected state [slots, nheads, 128, 128] and x [batch, nheads, 128]")
    if nheads % ngroups:
        raise ValueError("nheads must be divisible by ngroups")
    heads_per_group = nheads // ngroups
    if (head_tiles != (heads_per_group + 3) // 4
            or total_head_tiles != batch * ngroups * head_tiles):
        raise ValueError("work tiles must cover every (batch, group, head tile)")
    if batch != 1:
        raise ValueError("binding covers one batch row (one destination slot)")

    def slot_of(indices):
        # (int)slot is the TMA coordinate; TMA bounds-checks it against `slots`.
        slot = indices.to(torch.int32).to(torch.int64)
        valid = (slot >= 0) & (slot < slots)
        return slot, valid, torch.clamp(slot, 0, slots - 1)

    _, src_valid, src_safe = slot_of(state_batch_indices)
    initial = torch.index_select(state, 0, src_safe).float()
    s0 = torch.where(src_valid.reshape(batch, 1, 1, 1), initial,
                     torch.zeros_like(initial))

    delta = dt.float() + dt_bias.float()
    if dt_softplus:
        delta = torch.where(delta <= 20.0, torch.log(1.0 + torch.exp(delta)), delta)
    decay = torch.exp(A.float() * delta)
    b = torch.stack([B[:, h // heads_per_group, :] for h in range(nheads)], dim=1).float()
    c = torch.stack([C[:, h // heads_per_group, :] for h in range(nheads)], dim=1).float()
    xf = x.float()
    s1 = s0 * decay[:, :, None, None] + (
        b[:, :, None, :] * delta[:, :, None, None]) * xf[:, :, :, None]
    y = (s1 * c[:, :, None, :]).sum(dim=-1) + D.float()[None, :, None] * xf
    output = y.to(torch.bfloat16)

    dst, dst_valid, _ = slot_of(dst_state_batch_indices)
    rows = torch.arange(slots, device=state.device)
    hit = (rows == dst[0]) & dst_valid[0]
    new_state = torch.where(hit.reshape(slots, 1, 1, 1),
                            s1[0].to(torch.bfloat16), state)
    return output, new_state
