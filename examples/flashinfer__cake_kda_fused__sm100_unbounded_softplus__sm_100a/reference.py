"""Cake fused M128 KDA prefill, unbounded-softplus gate, one packed sequence and one head.

Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/kda/cake_kda_bf16_fused_m128_unbounded_softplus.cu
  748-768 task / sequence / slot selection (slot = seq_order[task], use_state_indices = 0),
  803-845 initial state (zero when use_initial_state = 0), 847-880 checkpoint row 0,
  890-1030 chunk loop, 992-1026 checkpoint rows before tokens 128c, 1033-1062 final state,
  1375-1376 / 1431-1442 / 1491-1515 gate, 1420-1426 and 1466-1488 beta: full 32-token chunks
  read beta_tma (TMA, line 1402), the partial tail chunk reads beta (line 1476-1478); both
  apply sigmoid and round the result to bf16 (line 1486-1488), 1665-1668 q/k L2 norm.
Host: csrc/kda/cake_kda_bf16_fused_m128_unbounded_softplus_binding.cu L44-121,
  csrc/kda/cake_kda_binding_common.cuh L488-506 (beta_tma = head-padded copy of beta),
  L506-714 (TMA maps); flashinfer/kda.py L108-262 (contract).

Per token t of the sequence:
  q' = q*scale*rsqrt(sum q^2 + 1e-6), k' = k*rsqrt(sum k^2 + 1e-6)
  decay = exp(-exp(A_log) * softplus(g + dt_bias)), softplus(x) = x for x > 20
  b = bf16(sigmoid(beta_t)), beta_t from beta_padded on full chunks and beta on the tail
  S' = S*decay ; u = (v - S'k')*b ; S = S' + u k'^T ; out_t = S q'
Checkpoint row c = state before token 128c (row 0 = zero initial state); final state bf16.
"""
import torch


def reference(q, k, v, g, beta, beta_padded, A_log, dt_bias,
              seq_start, seq_end, checkpoint_start, scale, checkpoint_every):
    total_tokens, heads, dim = q.shape
    if heads != 1 or seq_start != 0 or seq_end != total_tokens or checkpoint_start != 0:
        raise ValueError("binding covers one sequence spanning the packed tokens, one head")
    tokens = seq_end - seq_start
    full_tokens = (tokens // 32) * 32            # tokens in full 32-token chunks

    qf = q.float()
    kf = k.float()
    qn = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6) * scale
    kn = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)
    biased = g.float() + dt_bias.float()
    softplus = torch.where(biased <= 20.0, torch.log(1.0 + torch.exp(biased)), biased)
    decay = torch.exp(-torch.exp(A_log.float()).unsqueeze(-1) * softplus)     # [T, H, K]
    logit = torch.cat((beta_padded[:full_tokens, :heads], beta[full_tokens:tokens]), dim=0).float()
    b = (1.0 / (1.0 + torch.exp(-logit))).to(torch.bfloat16).float()          # [T, H]
    vf = v.float()

    state = torch.zeros((heads, dim, dim), dtype=torch.float32, device=q.device)   # [H, V, K]
    outputs, checkpoints = [], []
    for t in range(tokens):
        if t % checkpoint_every == 0:
            checkpoints.append(state)
        decayed = state * decay[t].unsqueeze(1)
        prediction = (decayed * kn[t].unsqueeze(1)).sum(dim=-1)              # [H, V]
        update = (vf[t] - prediction) * b[t].unsqueeze(-1)
        state = decayed + update.unsqueeze(-1) * kn[t].unsqueeze(1)
        outputs.append((state * qn[t].unsqueeze(1)).sum(dim=-1))
    out = torch.stack(outputs, dim=0).to(torch.bfloat16)                      # [T, H, V]
    final = state.unsqueeze(0).to(torch.bfloat16)                             # [1, H, V, K]
    return out, final, torch.stack(checkpoints, dim=0).to(torch.bfloat16)
