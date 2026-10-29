"""Persistent FlashKDA M128 prefill (lower-bound gate) with an in-place state pool.

Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/kda/cake_flashkda_bf16_persistent_m128.cu
  746-757 persistent task walk (task_offsets[bid] .. task_offsets[bid+1], task_ids, seq_order,
  cu_seqlens; head = task % H, state slot = seq_idx), 793-830 initial state read (use_initial_state),
  927-956 in-place final state store through initial_state, 1374-1386 gate rate exp(A_log) and bias
  exp(A_log)*dt_bias, 1447-1449 / 1511-1517 beta sigmoid (tanh form), 1452-1466 gate
  log2 = lower_bound*log2(e)*sigmoid(exp(A_log)*(g+dt_bias)), 1752-1754 q/k L2 norm.
Host: csrc/kda/cake_flashkda_bf16_persistent_m128_binding.cu L46-118 (in-place state required);
  flashinfer/kda.py L108-262 (contract).

task_ids and seq_order are permutations (input.json tensor_properties: injective, in range), so every
(sequence, head) task runs exactly once and the result does not depend on their contents.
Per sequence s, head h and token t in [cu_seqlens[s], cu_seqlens[s+1]):
  q' = q*scale*rsqrt(sum q^2 + 1e-6), k' = k*rsqrt(sum k^2 + 1e-6)
  decay = exp(lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias[h])))
  S' = S*decay ; u = (v - S'k')*sigmoid(beta) ; S = S' + u k'^T ; out_t = S q'
The state pool slot s starts from state[s] and is replaced by bf16(S) after the sequence.
"""
import torch


def reference(q, k, v, g, beta, A_log, dt_bias, state,
              cu_start, cu_mid, cu_end, scale, lower_bound):
    total_tokens, heads, dim = q.shape
    bounds = [cu_start, cu_mid, cu_end]
    if state.shape[0] != len(bounds) - 1 or cu_start != 0 or cu_end != total_tokens:
        raise ValueError("binding covers two packed sequences, one state slot each")
    qf = q.float()
    kf = k.float()
    qn = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6) * scale
    kn = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)
    argument = torch.exp(A_log.float()).unsqueeze(-1) * (g.float() + dt_bias.float())
    decay = torch.exp(lower_bound / (1.0 + torch.exp(-argument)))             # [T, H, K]
    b = 1.0 / (1.0 + torch.exp(-beta.float()))                                # [T, H]
    vf = v.float()

    outputs, finals = [], []
    for s in range(len(bounds) - 1):
        current = state[s].float()                                            # [H, V, K]
        for t in range(bounds[s], bounds[s + 1]):
            decayed = current * decay[t].unsqueeze(1)
            prediction = (decayed * kn[t].unsqueeze(1)).sum(dim=-1)          # [H, V]
            update = (vf[t] - prediction) * b[t].unsqueeze(-1)
            current = decayed + update.unsqueeze(-1) * kn[t].unsqueeze(1)
            outputs.append((current * qn[t].unsqueeze(1)).sum(dim=-1))
        finals.append(current)
    out = torch.stack(outputs, dim=0).to(torch.bfloat16)                      # [T, H, V]
    new_state = torch.stack(finals, dim=0).to(torch.bfloat16)                 # [N, H, V, K]
    return out, new_state
