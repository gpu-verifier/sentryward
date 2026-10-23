"""Packed T=1 KDA decode (Kimi K3 serving schedule), register tile8 interleaved.

Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/kda/cake_kda_packed_t1_register_tile8_interleaved.cu
  lines 134-612 (kernel_flashinfer_packed_kda_t1_register_tile8_interleaved body)
Layout contract: csrc/kda/cake_kda_packed_t1_binding.cuh lines 80-83, 160-276
  (H = 12 heads, D = 128, state [slots, 12, V=128, K=128], raw_beta [B, 12], out [B, 12, 128]).

Per batch row n and head h (no GQA, H = HV = 12):
  q' = q * scale * rsqrt(sum q^2 + 1e-6), k' = k * rsqrt(sum k^2 + 1e-6)     (lines 217-274)
  decay = exp(-5 * sigmoid(exp(A_log[h]) * (g + dt_bias[h])))                (lines 275-297)
  beta' = sigmoid(beta)                                                      (lines 328-336)
  S' = S * decay ; delta = (v - S' k') * beta' ; S_new = S' + delta k'^T   (lines 345-554)
  out = S_new q'  (fp32 S_new, before the bf16 state store)                 (lines 542-605)
A negative slot leaves state untouched and writes out = 0 (lines 154-159, 606-611).
State slots not selected by an active row are returned unchanged.
"""
import torch


def reference(q, k, v, g, beta, A_log, dt_bias, state, state_indices, scale):
    batch, heads, dim = q.shape
    slots = state.shape[0]
    if state_indices.shape[0] != batch:
        raise ValueError("state_indices must have one entry per batch row")
    qf, kf, vf = q.float(), k.float(), v.float()
    qs = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6) * scale     # [B, H, K]
    ks = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)             # [B, H, K]
    biased = g.float() + dt_bias.float()                                          # [B, H, K]
    gate_sigmoid = 1.0 / (1.0 + torch.exp(-torch.exp(A_log.float()).unsqueeze(-1) * biased))
    decay = torch.exp(-5.0 * gate_sigmoid)                                        # lower bound fixed at -5 in the source
    b = 1.0 / (1.0 + torch.exp(-beta.float()))                                    # [B, H]

    raw_slot = state_indices.to(torch.int64)                                      # [B]
    active = raw_slot >= 0
    safe_slot = torch.where(active, raw_slot, torch.zeros_like(raw_slot))
    initial = state.index_select(0, safe_slot).float()                            # [B, H, V, K]

    decayed = initial * decay.unsqueeze(2)
    prediction = (decayed * ks.unsqueeze(2)).sum(dim=-1)                          # [B, H, V]
    delta = (vf - prediction) * b.unsqueeze(-1)
    updated = decayed + delta.unsqueeze(-1) * ks.unsqueeze(2)                     # [B, H, V, K]
    o = (updated * qs.unsqueeze(2)).sum(dim=-1)                                   # [B, H, V]

    out = torch.where(active.reshape(batch, 1, 1), o, torch.zeros_like(o)).to(torch.bfloat16)
    # Active slots are unique (caller contract; input.json states it as `injective`), so the
    # per-row writes commute; rows are applied in order with torch.where on an arange comparison.
    slot_ids = torch.arange(slots, device=state.device)
    stored = updated.to(torch.bfloat16)
    new_state = state
    for n in range(batch):
        selected = (slot_ids == raw_slot[n]) & active[n]
        new_state = torch.where(selected.reshape(slots, 1, 1, 1), stored[n].unsqueeze(0), new_state)
    return new_state, out
