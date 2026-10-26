"""Single-token Mamba selective state update (FlashInfer simple kernel).

Source: flashinfer-ai/flashinfer b250399302acb2c4925b03416188c6678b762fbd,
include/flashinfer/mamba/kernel_selective_state_update_stp.cuh:77-306
(selective_state_update_kernel_simple<bf16, bf16, float, bf16, int64, void,
DIM=64, DSTATE=128, ROWS_PER_BLOCK=4, numWarps=4, PHILOX_ROUNDS=0>),
params struct include/flashinfer/mamba/selective_state_update.cuh:51-87,
softplus include/flashinfer/mamba/common.cuh:85-90.

The state is read from slot state_batch_indices[b] (zero when that index
equals pad_slot_id) and the new state is written to slot
dst_state_batch_indices[b] when update_state is set and the source slot is
not the pad slot. The returned state is the new content of every slot.
"""
import torch


def reference(state, x, dt, dt_bias, A, B, C, D, z,
              state_batch_indices, dst_state_batch_indices,
              nheads, ngroups, pad_slot_id, dt_softplus, update_state):
    batch, heads, dim = x.shape
    slots, state_heads, state_dim, dstate = state.shape
    if heads != nheads or state_heads != nheads or state_dim != dim:
        raise ValueError("state/x shapes disagree with nheads and dim")
    if ngroups <= 0 or nheads % ngroups:
        raise ValueError("nheads must be divisible by ngroups")
    per_group = nheads // ngroups

    # Source slot; the pad sentinel reads a zero state (no load).
    source = state_batch_indices
    valid = source != pad_slot_id
    initial = torch.index_select(state, 0, source).float()
    s = torch.where(valid.reshape(batch, 1, 1, 1), initial, 0.0)

    # dt (bf16) + dt_bias (bf16), thresholded softplus, decay exp(A * dt).
    delta = dt.float() + dt_bias.float()
    if dt_softplus:
        delta = torch.where(delta <= 20.0,
                            torch.log(1.0 + torch.exp(delta)), delta)
    decay = torch.exp(A.float() * delta)

    # Each head uses its group's B/C vector.
    b = torch.stack([B[:, head // per_group, :] for head in range(nheads)],
                    dim=1).float()
    c = torch.stack([C[:, head // per_group, :] for head in range(nheads)],
                    dim=1).float()
    xv = x.float()
    new_state = s * decay.reshape(batch, nheads, 1, 1) + (
        b.reshape(batch, nheads, 1, dstate) * delta.reshape(batch, nheads, 1, 1)
    ) * xv.reshape(batch, nheads, dim, 1)

    # Output uses the unrounded FP32 state; z gate is silu(z).
    y = (new_state * c.reshape(batch, nheads, 1, dstate)).sum(dim=-1)
    y = y + D.float().reshape(1, nheads, 1) * xv
    zv = z.float()
    y = y * (zv * torch.reciprocal(1.0 + torch.exp(-zv)))
    output = y.to(x.dtype)

    # Stored state is the BF16 rounding of the FP32 new state, written to the
    # destination slot only; every other slot keeps its initial contents.
    stored = new_state.to(state.dtype)
    slot_ids = torch.arange(slots, device=state.device)
    result = state.clone()
    if update_state:
        for sample in range(batch):
            hit = (slot_ids == dst_state_batch_indices[sample]) & valid[sample]
            result = torch.where(hit.reshape(slots, 1, 1, 1), stored[sample], result)
    return output, result
