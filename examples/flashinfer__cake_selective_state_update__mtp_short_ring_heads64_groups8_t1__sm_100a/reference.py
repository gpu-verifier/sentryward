"""Short (one-token) BF16 selective state update, 64 heads of 8 groups (CAKE mtp_short_ring_heads64_groups8_t1).

Source (flashinfer 47e64b0e96c446eaba90e40a1ecdf5891aaef92d):
  csrc/cake_selective_state_update/cake_selective_state_update_141341c762041c2f0989_kernel.cu
    lines 173-653 (kernel_cake_selective_state_update_141341c762041c2f0989: one CTA per (batch row, group),
    eight heads per CTA, a 12-stage TMA ring of 32-row state stages; state read and written in place through
    the tensor map at the slot of state_batch_indices)
  csrc/cake_selective_state_update/cake_selective_state_update_141341c762041c2f0989_binding.cu
    lines 70-97 (4-D state tensor map, box {128, 32, 1, 1}, OOB fill none = zeros), 165-174 (dim == dstate ==
    128, nheads == 64, ngroups == 8, token_steps == 1), 212-213 (288 threads, 103680 B)
  flashinfer/jit/mamba/cake_selective_state_update.py lines 2799-2827 (mtp_short_program), 3845-3904 (short
    route: x/output [batch, T, nheads, 128], B/C [batch, T, ngroups, 128], compact dt [batch, T, nheads],
    A/D/dt_bias [nheads], no destination table, in-place update, grid batch * ngroups * head_tiles)
  PR #6030 (thresholded softplus in the short programs: fminf(dt, 20) before expf, max after log1pf)
  tests/mamba/test_cake_selective_state_update.py (parity with selective_state_update(backend="flashinfer"))

Per batch row b, head h (group g = h // (nheads // ngroups)), token t, row r, column s:
  u_t  = dt[b, t, h] + dt_bias[h]
  dt_t = max(log1p(exp(min(u_t, 20))), u_t) when dt_softplus, else u_t
  S    = S * exp(A[h] * dt_t) + B[b, t, g, s] * (dt_t * x[b, t, h, r])
  y_t  = sum_s S[r, s] * C[b, t, g, s] + D[h] * x[b, t, h, r]       (fp32 S)
The slot's state [nheads, 128, 128] is read through the tensor map before token 0 and written once (bf16) after
the last token at the same slot.  The int64 slot is passed as the int32 tensor-map coordinate: an out-of-range
coordinate loads zeros and its store is dropped.  slot == pad_slot_id: start from zero state and do not write
the state.  Slots no row names keep their contents.  When several rows name one slot the last row is taken
here; the kernel would race there, which this reference does not model.
"""
import torch


def reference(state, x, dt, A, B, C, D, dt_bias, state_batch_indices,
              batch_size, nheads, dim, dstate, ngroups, token_steps, head_tiles,
              dt_softplus, pad_slot_id, grid_x):
    slots, heads, rows, cols = state.shape
    batch, steps, x_heads, width = x.shape
    if (batch, steps, x_heads, width) != (batch_size, token_steps, nheads, 128):
        raise ValueError("x must be [batch_size, token_steps, nheads, 128]")
    if (heads, rows, cols) != (nheads, 128, 128) or (dim, dstate) != (128, 128):
        raise ValueError("expected dim = dstate = 128 and state [slots, nheads, 128, 128]")
    if token_steps not in (1, 2):
        raise ValueError("expected 1 <= token_steps <= 2")
    if nheads % ngroups:
        raise ValueError("nheads must be divisible by ngroups")
    heads_per_group = nheads // ngroups
    if head_tiles != -(-heads_per_group // 8) or grid_x != batch_size * ngroups * head_tiles:
        raise ValueError("launch must cover every (batch row, group, eight-head tile)")
    if tuple(B.shape) != (batch, steps, ngroups, 128) or tuple(C.shape) != (batch, steps, ngroups, 128) or \
            tuple(dt.shape) != (batch, steps, nheads):
        raise ValueError("expected B/C [batch, T, ngroups, 128] and compact dt [batch, T, nheads]")
    # pad_slot_id is a long long compared with the int64 slot; the u64 record holds its bits.
    pad = pad_slot_id - (1 << 64) if pad_slot_id >= (1 << 63) else pad_slot_id

    slot = state_batch_indices[:batch]
    coord = slot.to(torch.int32).to(torch.int64)
    in_range = (coord >= 0) & (coord < slots)
    active = slot != pad
    initial = torch.index_select(state, 0, torch.clamp(coord, 0, slots - 1)).float()
    live = (active & in_range).reshape(batch, 1, 1, 1)
    # (batch, group, head-in-group, row, column) view: heads of one group share that group's B / C row.
    g5 = (batch, ngroups, heads_per_group, 128, 128)
    s = torch.where(live, initial, torch.zeros_like(initial)).reshape(g5)

    outputs = []
    for t in range(token_steps):
        delta = dt[:, t, :].float() + dt_bias.float().reshape(1, nheads)
        if dt_softplus:
            # max(log1p(exp(min(u, 20))), u), spelled with where/log for reference capture.
            softplus = torch.log(1.0 + torch.exp(torch.clamp(delta, max=20.0)))
            delta = torch.where(softplus >= delta, softplus, delta)
        decay = torch.exp(A.float().reshape(1, nheads) * delta)
        b5 = B[:, t, :, :].float().reshape(batch, ngroups, 1, 1, 128)
        c5 = C[:, t, :, :].float().reshape(batch, ngroups, 1, 1, 128)
        x5 = x[:, t, :, :].float().reshape(batch, ngroups, heads_per_group, 128, 1)
        delta5 = delta.reshape(batch, ngroups, heads_per_group, 1, 1)
        decay5 = decay.reshape(batch, ngroups, heads_per_group, 1, 1)
        s = s * decay5 + b5 * (delta5 * x5)
        y = (s * c5).sum(dim=-1) + D.float().reshape(1, ngroups, heads_per_group, 1) * x5.reshape(
            batch, ngroups, heads_per_group, 128)
        outputs.append(y.reshape(batch, nheads, 128))
    output = torch.stack(outputs, dim=1).to(torch.bfloat16)

    stored = s.reshape(batch, nheads, 128, 128).to(torch.bfloat16)
    store = active & in_range
    positions = torch.arange(slots, device=state.device)
    new_state = state
    for b in range(batch):
        hit = (positions == coord[b]) & store[b]
        new_state = torch.where(hit.reshape(slots, 1, 1, 1), stored[b], new_state)
    return output, new_state
