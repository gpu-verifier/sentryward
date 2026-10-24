"""One- or two-token bf16 selective state update (CAKE mtp_short), state updated in place.

Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/cake_selective_state_update/cuda/cake_selective_state_update_mtp_short.cu
  lines 126-1282 (kernel_cake_selective_state_update_mtp_short)
Host/dispatch: csrc/cake_selective_state_update/host/cake_selective_state_update_mtp_short.cc
  lines 163-258 (state [slots, nheads, 128, 128], x/output [batch, T, nheads, 128],
  B/C [batch, T, ngroups, 128], dt [batch, T, nheads], dim == dstate == 128, 1 <= T <= 2,
  grid = batch * nheads); flashinfer/jit/mamba/cake_selective_state_update.py lines 584-628
  (state_stride_slot = state.stride(0), no destination table, in-place update).

Per head h (group g = h // (nheads // ngroups)), token t, row r, column s:
  dt_t = softplus(dt[b, t, h] + dt_bias[h]) = log1p(exp(.)) (no threshold), when dt_softplus
  S    = S * exp(A[h] * dt_t) + B[b, t, g, s] * dt_t * x[b, t, h, r]
  y_t  = sum_s S[r, s] * C[b, t, g, s] + D[h] * x[b, t, h, r]       (fp32 S)
The slot's state [nheads, 128, 128] is read before token 0 and written once (bf16) after
the last token. slot == pad_slot_id: start from zero state and do not write the state.
Every other slot keeps its contents.
"""
import torch


def reference(state, x, dt, A, B, C, D, dt_bias, state_batch_indices,
              batch_size, nheads, dim, dstate, ngroups, token_steps,
              state_stride_slot, dt_softplus, pad_slot_id, grid_x):
    slots, heads, rows, cols = state.shape
    batch, steps, x_heads, width = x.shape
    if (batch, steps, x_heads, width) != (batch_size, token_steps, nheads, 128):
        raise ValueError("x must be [batch_size, token_steps, nheads, 128]")
    if (heads, rows, cols) != (nheads, 128, 128) or (dim, dstate) != (128, 128):
        raise ValueError("expected dim = dstate = 128 and state [slots, nheads, 128, 128]")
    if state_stride_slot != nheads * 128 * 128 or token_steps not in (1, 2):
        raise ValueError("expected a dense slot stride and 1 <= token_steps <= 2")
    if nheads % ngroups or grid_x != batch_size * nheads:
        raise ValueError("launch must cover every (batch, head) pair")
    if batch_size != 1:
        raise ValueError("binding covers one batch row (one state slot)")
    # pad_slot_id is a long long compared with the int64 slot; the u64 record holds its bits.
    pad = pad_slot_id - (1 << 64) if pad_slot_id >= (1 << 63) else pad_slot_id
    heads_per_group = nheads // ngroups

    slot = state_batch_indices[0]
    active = slot != pad
    safe = torch.where(active, slot, torch.zeros_like(slot))
    initial = torch.index_select(state, 0, safe.reshape(1))[0].float()
    s = torch.where(active, initial, torch.zeros_like(initial))

    outputs = []
    for t in range(token_steps):
        delta = dt[0, t, :].float() + dt_bias.float()
        if dt_softplus:
            delta = torch.log(1.0 + torch.exp(delta))
        decay = torch.exp(A.float() * delta)
        b = torch.stack([B[0, t, h // heads_per_group, :] for h in range(nheads)], dim=0).float()
        c = torch.stack([C[0, t, h // heads_per_group, :] for h in range(nheads)], dim=0).float()
        xt = x[0, t, :, :].float()
        s = s * decay[:, None, None] + (b[:, None, :] * delta[:, None, None]) * xt[:, :, None]
        outputs.append((s * c[:, None, :]).sum(dim=-1) + D.float()[:, None] * xt)
    output = torch.stack(outputs, dim=0).reshape(1, token_steps, nheads, 128).to(torch.bfloat16)

    hit = (torch.arange(slots, device=state.device) == slot) & active
    new_state = torch.where(hit.reshape(slots, 1, 1, 1), s.to(torch.bfloat16), state)
    return output, new_state
