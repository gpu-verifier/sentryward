"""Six-token BF16 selective state update with cached intermediate states (CAKE mtp_horizontal).

Source (flashinfer b250399302acb2c4925b03416188c6678b762fbd):
  csrc/cake_selective_state_update/cuda/cake_selective_state_update_mtp_horizontal.cu
    lines 269-627 (kernel_cake_selective_state_update_mtp_horizontal: persistent CTAs over work groups of two
    (batch row, head) tiles; the 64 x 128 state of the source slot read through the tensor map; six tokens in
    fp32; the state after each token stored to the intermediate buffer; the state itself is never written)
  csrc/cake_selective_state_update/host/cake_selective_state_update_mtp_horizontal.cc
    lines 205-247 (4-D state tensor map, box {128, 32, 1, 1}, OOB fill none = zeros), 379-495 (launch)
  flashinfer/jit/mamba/cake_selective_state_update.py lines 630-727 (horizontal route: compact dt
    [batch, T, nheads], A/D/dt_bias [nheads], total_tiles = batch * nheads, intermediate stride = stride(0))
  flashinfer/mamba/selective_state_update.py line 304 (output = torch.empty_like(x))
  tests/mamba/test_cake_selective_state_update.py lines 239-355 (row mtp_horizontal; parity with
    selective_state_update(backend="flashinfer") on output, state and intermediate_states_buffer)

Per batch row b, head h (group g = h // (nheads // ngroups)), token t, row r, column s:
  u_t  = dt[b, t, h] + dt_bias[h]
  dt_t = u_t <= 20 ? log(1 + exp(u_t)) : u_t when dt_softplus, else u_t
  S    = S * exp(A[h] * dt_t) + (B[b, t, g, s] * dt_t) * x[b, t, h, r]
  intermediate_state[cache[b], t, h, r, s] = bf16(S) when cache_intermediate and src[b] >= 0
  y_t  = sum_s S[r, s] * C[b, t, g, s] + D[h] * x[b, t, h, r]       (fp32 S)
S starts from state[src[b], h]: the int64 slot is passed as the int32 tensor-map coordinate, and an
out-of-range coordinate loads zeros.  cache[b] = intermediate_state_indices[b] selects the intermediate slot.
Buffer slots no stored row names keep their contents.  When several rows name one cache slot the last row is
taken here; the kernel would race there, which this reference does not model.  A cache slot outside the
buffer is an out-of-bounds store in the kernel; this reference does not model it.
"""
import torch

TOKENS = 6
DIM = 64
DSTATE = 128


def reference(state, x, B, C, dt, A, D, dt_bias, state_batch_indices, intermediate_state,
              intermediate_state_indices, nheads, ngroups, total_tiles, intermediate_stride_slot,
              dt_softplus, cache_intermediate):
    slots, heads, rows, cols = state.shape
    batch, tokens, x_heads, width = x.shape
    if (tokens, x_heads, width) != (TOKENS, nheads, DIM) or (heads, rows, cols) != (nheads, DIM, DSTATE):
        raise ValueError("expected x [batch, 6, nheads, 64] and state [slots, nheads, 64, 128]")
    if nheads % ngroups:
        raise ValueError("nheads must be divisible by ngroups")
    if total_tiles != batch * nheads:
        raise ValueError("total_tiles must cover every (batch row, head) tile")
    if tuple(B.shape) != (batch, TOKENS, ngroups, DSTATE) or tuple(C.shape) != tuple(B.shape) or \
            tuple(dt.shape) != (batch, TOKENS, nheads):
        raise ValueError("expected B/C [batch, 6, ngroups, 128] and compact dt [batch, 6, nheads]")
    cache_slots = intermediate_state.shape[0]
    if tuple(intermediate_state.shape[1:]) != (TOKENS, nheads, DIM, DSTATE) or \
            intermediate_stride_slot != TOKENS * nheads * DIM * DSTATE:
        raise ValueError("expected a dense intermediate buffer [slots, 6, nheads, 64, 128]")
    heads_per_group = nheads // ngroups

    src = state_batch_indices[:batch]
    cache = intermediate_state_indices[:batch]
    # PTX converts the int64 slot to the signed int32 TMA coordinate.
    # TMA zero-fills out-of-bounds slots; negative indices do not wrap.
    coord = src.to(torch.int32).to(torch.int64)
    in_range = (coord >= 0) & (coord < slots)
    initial = torch.index_select(state, 0, torch.clamp(coord, 0, slots - 1)).float()
    # (batch, group, head-in-group, row, column) view: heads of one group share that group's B / C row.
    g5 = (batch, ngroups, heads_per_group, DIM, DSTATE)
    s = torch.where(in_range.reshape(batch, 1, 1, 1), initial, torch.zeros_like(initial)).reshape(g5)

    outputs = []
    states = []
    for t in range(TOKENS):
        delta = dt[:, t, :].float() + dt_bias.float().reshape(1, nheads)
        if dt_softplus:
            delta = torch.where(delta <= 20.0, torch.log(1.0 + torch.exp(delta)), delta)
        decay = torch.exp(A.float().reshape(1, nheads) * delta)
        b5 = B[:, t, :, :].float().reshape(batch, ngroups, 1, 1, DSTATE)
        c5 = C[:, t, :, :].float().reshape(batch, ngroups, 1, 1, DSTATE)
        x5 = x[:, t, :, :].float().reshape(batch, ngroups, heads_per_group, DIM, 1)
        delta5 = delta.reshape(batch, ngroups, heads_per_group, 1, 1)
        decay5 = decay.reshape(batch, ngroups, heads_per_group, 1, 1)
        s = s * decay5 + (b5 * delta5) * x5
        states.append(s.reshape(batch, nheads, DIM, DSTATE))
        y = (s * c5).sum(dim=-1) + D.float().reshape(1, ngroups, heads_per_group, 1) * x5.reshape(
            batch, ngroups, heads_per_group, DIM)
        outputs.append(y.reshape(batch, nheads, DIM))
    output = torch.stack(outputs, dim=1).to(torch.bfloat16)

    new_cache = intermediate_state
    if cache_intermediate:
        stored = torch.stack(states, dim=1).to(torch.bfloat16)
        positions = torch.arange(cache_slots, device=state.device)
        for b in range(batch):
            hit = (positions == cache[b]) & (src[b] >= 0)
            new_cache = torch.where(hit.reshape(cache_slots, 1, 1, 1, 1), stored[b], new_cache)
    return output, new_cache
