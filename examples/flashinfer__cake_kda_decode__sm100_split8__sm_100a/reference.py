"""Recurrent KDA single-token decode, unbounded-softplus gate, split8 value tiling.

Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/kda/cake_kda_decode_d128_t1_unbounded_softplus_direct_split8.cu
  lines 63-317 (kernel_flashinfer_recurrent_kda_t1_unbounded_softplus body)
Layout contract: csrc/kda/cake_kda_decode_binding_common.cuh lines 142-188, 271-327
  (q/k [1,T,H,128], v/g [1,T,HV,128], beta [1,T,HV], state [slots,HV,V=128,K=128]);
  flashinfer/kda_decode.py lines 98-115 (A_log [H], dt_bias [H*K]).

Per value head hv (query head hv // head_ratio), for the token of sequence 0:
  q' = q * scale * rsqrt(sum q^2 + 1e-6), k' = k * rsqrt(sum k^2 + 1e-6)
  sp = softplus(g + dt_bias) with the source's x > 20 -> x passthrough
  decay = exp(-exp(A_log) * sp)
  S' = S * decay ; delta = (v - S' k') * beta ; S_new = S' + delta k'^T
  out = S' q' + delta (k'.q')        (== S_new q')
A negative slot index means: read slot 0, do not write state, write out = 0.
State slots other than the selected one are returned unchanged.
"""
import torch


def reference(q, k, v, g, beta, A_log, dt_bias, state, ssm_state_indices,
              seq_start, seq_end, scale, beta_is_logit, H, HV, head_ratio):
    _, heads, dim = q.shape
    slots, value_heads, value_dim, key_dim = state.shape
    if heads != H or value_heads != HV or H * head_ratio != HV or ssm_state_indices.shape[0] != 1:
        raise ValueError("binding covers one sequence with HV = H * head_ratio")
    if seq_end - seq_start <= 0:
        raise ValueError("binding covers a nonempty single-token sequence")
    t = seq_start

    qh = torch.stack([q[t, hv // head_ratio, :] for hv in range(HV)], dim=0).float()
    kh = torch.stack([k[t, hv // head_ratio, :] for hv in range(HV)], dim=0).float()
    a = torch.stack([A_log[hv // head_ratio] for hv in range(HV)], dim=0).float()
    bias = torch.stack([dt_bias[hv // head_ratio, :] for hv in range(HV)], dim=0).float()
    gate = g[t].float()                      # [HV, K]
    value = v[t].float()                     # [HV, V]
    b = beta[t].float()                      # [HV]
    if beta_is_logit:
        b = 1.0 / (1.0 + torch.exp(-b))

    qs = qh * torch.rsqrt((qh * qh).sum(dim=-1, keepdim=True) + 1e-6) * scale
    ks = kh * torch.rsqrt((kh * kh).sum(dim=-1, keepdim=True) + 1e-6)
    biased = gate + bias
    softplus = torch.where(biased <= 20.0, torch.log(1.0 + torch.exp(biased)), biased)
    decay = torch.exp(-torch.exp(a).unsqueeze(-1) * softplus)          # [HV, K]
    k_dot_q = (ks * qs).sum(dim=-1)                                     # [HV]

    raw_slot = ssm_state_indices[0].to(torch.int64)
    active = raw_slot >= 0
    initial_slot = torch.where(active, raw_slot, torch.zeros_like(raw_slot))
    initial = state.index_select(0, initial_slot.reshape(1))[0].float()  # [HV, V, K]

    decayed = initial * decay.unsqueeze(1)
    prediction = (decayed * ks.unsqueeze(1)).sum(dim=-1)                # [HV, V]
    base = (decayed * qs.unsqueeze(1)).sum(dim=-1)                      # [HV, V]
    delta = (value - prediction) * b.unsqueeze(-1)
    updated = decayed + delta.unsqueeze(-1) * ks.unsqueeze(1)           # [HV, V, K]
    o = base + delta * k_dot_q.unsqueeze(-1)                            # [HV, V]

    out = torch.where(active, o, torch.zeros_like(o)).to(torch.bfloat16).reshape(1, HV, value_dim)
    selected = (torch.arange(slots, device=state.device) == raw_slot) & active
    new_state = torch.where(selected.reshape(slots, 1, 1, 1),
                            updated.to(torch.bfloat16).unsqueeze(0), state)
    return new_state, out
