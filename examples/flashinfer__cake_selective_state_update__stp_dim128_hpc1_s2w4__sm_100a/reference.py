"""Single-token BF16 selective state update, headdim 128 / dstate 128 (CAKE stp_dim128_hpc1_s2w4).

Source (flashinfer 47e64b0e96c446eaba90e40a1ecdf5891aaef92d):
  csrc/cake_selective_state_update/cake_selective_state_update_6b20a536e14fc15a5a1b_kernel.cu
    lines 161-610 (kernel_cake_selective_state_update_6b20a536e14fc15a5a1b: one head per CTA, two CTAs per
    work tile, CTA bid & 1 owning state rows 64 * (bid & 1) .. + 63)
  csrc/cake_selective_state_update/cake_selective_state_update_6b20a536e14fc15a5a1b_binding.cu
    lines 70-97 (4-D state tensor map, box {128, 16, 1, 1}, OOB fill none = zeros), 197-198 (160 threads, 17536 B)
  flashinfer/jit/mamba/cake_selective_state_update.py lines 3138-3326 (dim128_split_program, _plan_stp_rows)
  flashinfer/mamba/selective_state_update.py (pad_slot_id default -1; the destination table is the source
    table when dst_state_batch_indices is None)
  tests/mamba/test_cake_selective_state_update.py (parity with selective_state_update(backend="flashinfer"))

Per batch row b and head h (group g = h // (nheads // ngroups)), state rows r and columns s:
  dt' = dt[b, h] + dt_bias[h]; with dt_softplus: dt' <= 20 ? log(1 + exp(dt')) : dt'
  S'  = S0 * exp(A[h] * dt') + B[b, g, s] * (dt' * x[b, h, r])
  y   = sum_s S'[r, s] * C[b, g, s] + D[h] * x[b, h, r]          (y uses the fp32 S')
  output[b, h, r] = bf16(y); state[dst[b], h] = bf16(S') when the row is stored.
S0 is state[src[b], h] read through the tensor map: the int64 slot is passed as the int32 coordinate, and an
out-of-range coordinate loads zeros.  A source slot equal to pad_slot_id starts from the zero state and stores
nothing; a destination slot equal to pad_slot_id stores nothing; disable_state_update stores nothing.  The store
goes through the tensor map at the int32 destination coordinate (an out-of-range coordinate is not written).
Slots no stored row names keep their contents.  When several rows name one destination slot the last row is
taken here; the kernel would race there, which this reference does not model.
"""
import torch


def reference(state, x, dt, A, B, C, D, dt_bias, state_batch_indices,
              dst_state_batch_indices, nheads, ngroups, head_tiles, x_batch_stride,
              b_batch_stride, c_batch_stride, out_batch_stride, dt_batch_stride,
              dt_softplus, has_z, disable_state_update, pad_slot_id, grid_x):
    if has_z:
        raise ValueError("binding covers has_z=0 only (no z tensor is bound)")
    slots, heads, dim, dstate = state.shape
    batch = x.shape[0]
    if heads != nheads or tuple(x.shape) != (batch, nheads, 128) or (dim, dstate) != (128, 128):
        raise ValueError("expected state [slots, nheads, 128, 128] and x [batch, nheads, 128]")
    if nheads % ngroups:
        raise ValueError("nheads must be divisible by ngroups")
    heads_per_group = nheads // ngroups
    # One head per CTA (HEADS_PER_CTA 1), two CTAs per (batch, group, head) tile (P-split 2).
    if head_tiles != heads_per_group or grid_x != batch * ngroups * head_tiles * 2:
        raise ValueError("launch must cover every (batch, group, head) tile with two row halves")
    if (x_batch_stride, out_batch_stride) != (nheads * 128, nheads * 128) or \
            (b_batch_stride, c_batch_stride) != (ngroups * 128, ngroups * 128) or \
            dt_batch_stride != nheads:
        raise ValueError("binding covers dense x/B/C/output rows and a dense [batch, nheads] dt")
    if tuple(B.shape) != (batch, ngroups, 128) or tuple(C.shape) != (batch, ngroups, 128) or \
            tuple(dt.shape) != (batch, nheads):
        raise ValueError("expected B/C [batch, ngroups, 128] and dt [batch, nheads]")
    # pad_slot_id is a long long compared with the int64 slots; the u64 record holds its bits.
    pad = pad_slot_id - (1 << 64) if pad_slot_id >= (1 << 63) else pad_slot_id

    src = state_batch_indices[:batch]
    dst = dst_state_batch_indices[:batch]
    src_coord = src.to(torch.int32).to(torch.int64)
    dst_coord = dst.to(torch.int32).to(torch.int64)
    src_live = (src != pad) & (src_coord >= 0) & (src_coord < slots)
    initial = torch.index_select(state, 0, torch.clamp(src_coord, 0, slots - 1)).float()
    s0 = torch.where(src_live.reshape(batch, 1, 1, 1), initial, torch.zeros_like(initial))

    delta = dt.float() + dt_bias.float().reshape(1, nheads)
    if dt_softplus:
        delta = torch.where(delta <= 20.0, torch.log(1.0 + torch.exp(delta)), delta)
    decay = torch.exp(A.float().reshape(1, nheads) * delta)
    # Heads of one group share that group's B / C row: work on a (batch, group, head-in-group) view.
    g5 = (batch, ngroups, heads_per_group, 128, 128)
    b5 = B.float().reshape(batch, ngroups, 1, 1, 128)
    c5 = C.float().reshape(batch, ngroups, 1, 1, 128)
    x5 = x.float().reshape(batch, ngroups, heads_per_group, 128, 1)
    delta5 = delta.reshape(batch, ngroups, heads_per_group, 1, 1)
    decay5 = decay.reshape(batch, ngroups, heads_per_group, 1, 1)
    s1 = s0.reshape(g5) * decay5 + b5 * (delta5 * x5)
    y = (s1 * c5).sum(dim=-1) + D.float().reshape(1, ngroups, heads_per_group, 1) * x5.reshape(
        batch, ngroups, heads_per_group, 128)
    output = y.reshape(batch, nheads, 128).to(torch.bfloat16)

    stored = s1.reshape(batch, nheads, 128, 128).to(torch.bfloat16)
    store = (src != pad) & (dst != pad) & (dst_coord >= 0) & (dst_coord < slots)
    if disable_state_update:
        store = store & False
    rows = torch.arange(slots, device=state.device)
    new_state = state
    for b in range(batch):
        hit = (rows == dst_coord[b]) & store[b]
        new_state = torch.where(hit.reshape(slots, 1, 1, 1), stored[b], new_state)
    return output, new_state
