"""Six-token selective state update; no state-index guarantees are assumed.

Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/cake_selective_state_update/cuda/cake_selective_state_update_mtp_horizontal.cu
"""
import torch


def reference(state, x, B, C, dt, A, D, dt_bias, state_batch_indices,
              nheads, ngroups, total_tiles, dt_softplus, cache_intermediate):
    if cache_intermediate:
        raise ValueError("This reference binding covers cache_intermediate=0 only")
    batch, tokens, heads, dim = x.shape
    if tokens != 6 or heads != nheads or total_tiles != batch * heads:
        raise ValueError("Expected six tokens and a complete batch/head tile set")
    if nheads % ngroups:
        raise ValueError("nheads must be divisible by ngroups")
    # PTX converts the int64 batch index to the signed int32 TMA coordinate.
    # TMA zero-fills out-of-bounds slots; negative indices do not wrap.
    slots = state_batch_indices.to(torch.int32).to(torch.int64)
    valid = (slots >= 0) & (slots < state.shape[0])
    safe_slots = torch.clamp(slots, 0, state.shape[0] - 1)
    initial = torch.index_select(state, 0, safe_slots).float()
    s = torch.where(valid.reshape(batch, 1, 1, 1), initial,
                    torch.zeros_like(initial))
    outputs = []
    for token in range(6):
        delta = dt[:, token, :].float() + dt_bias.float()
        if dt_softplus:
            delta = torch.where(delta <= 20.0,
                                torch.log(1.0 + torch.exp(delta)), delta)
        decay = torch.exp(A.float() * delta)
        # Each head uses its group's B/C vector.
        b = torch.stack([B[:, token, head // (nheads // ngroups), :]
                         for head in range(nheads)], dim=1).float()
        c = torch.stack([C[:, token, head // (nheads // ngroups), :]
                         for head in range(nheads)], dim=1).float()
        xt = x[:, token, :, :].float()
        s = s * decay[:, :, None, None] + (
            b[:, :, None, :] * delta[:, :, None, None]
        ) * xt[:, :, :, None]
        y = (s * c[:, :, None, :]).sum(dim=-1) + D.float()[None, :, None] * xt
        outputs.append(y)
    return torch.stack(outputs, dim=1).to(x.dtype)
