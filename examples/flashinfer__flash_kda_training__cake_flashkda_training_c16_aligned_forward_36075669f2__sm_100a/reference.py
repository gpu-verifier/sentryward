"""FlashKDA training forward (C16 aligned, checkpointed): packed sequences, H value heads.

Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/kda/cake_aligned_training_export/cake_flashkda_training_c16_aligned_forward_36075669f2.cu
  125-131 compile-time flags (USE_INITIAL_STATE 1, ENABLE_CHECKPOINTS 1, CHECKPOINT_STRIDE_CHUNKS 1,
  STORE_BETA_ACTIVE 1), 893-908 work item decode, gate rate exp(A_log[head]), bias dt_bias;
  925-945 beta_active = bf16(sigmoid(beta[token*H + head])) stored to beta_active_out[token*stride +
  head] and used by the recurrence; 960-976 gate log2 = lower_bound*log2(e)*sigmoid(rate*(g+bias));
  1110-1128 q/k inverse norms rsqrt(max(sum x^2, 1e-12)); 1386-1460 initial state (fp32, read for the
  first piece and rounded to bf16); 1466-1505 / 3040-3185 checkpoint row = bf16 state before every
  16-token chunk, at checkpoint_cu_starts[seq] + chunk; 2082-2100 output scaled by `scale`, stored bf16;
  2372-2412 fp32 final state for the last piece.
Host: csrc/kda/flashkda_training_paired_binding.cu L84 (kChunk 16), L96-140 (TMA maps), L229-357
  (RunTrainingForward: aligned variant when total tokens are a multiple of 16).

Per sequence s, head h and token t:
  q' = q*rsqrt(max(sum q^2, 1e-12)), k' = k*rsqrt(max(sum k^2, 1e-12))
  decay = exp(lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias[h])))
  b = bf16(sigmoid(beta)) ; S' = S*decay ; u = (v - S'k')*b ; S = S' + u k'^T ; out = scale * S q'
S starts from bf16(initial_state[s]).
"""
import torch


def reference(q, k, v, g, beta, A_log, dt_bias, initial_state,
              seq_len, cu_end, checkpoint_end, chunk, scale, lower_bound):
    total_tokens, heads, dim = q.shape
    sequences = initial_state.shape[0]
    # input.json fixes uniform sequences: cu_seqlens = [0, L, 2L, ...], checkpoint starts every L/16.
    if cu_end != sequences * seq_len or cu_end != total_tokens or seq_len % chunk \
            or checkpoint_end != sequences * (seq_len // chunk):
        raise ValueError("binding covers uniform, chunk-aligned packed sequences")
    bounds = [s * seq_len for s in range(sequences + 1)]
    qf = q.float()
    kf = k.float()
    qn = qf * torch.rsqrt(torch.clamp((qf * qf).sum(dim=-1, keepdim=True), min=1e-12))
    kn = kf * torch.rsqrt(torch.clamp((kf * kf).sum(dim=-1, keepdim=True), min=1e-12))
    argument = torch.exp(A_log.float()).unsqueeze(-1) * (g.float() + dt_bias.float())
    decay = torch.exp(lower_bound / (1.0 + torch.exp(-argument)))             # [T, H, K]
    beta_active = (1.0 / (1.0 + torch.exp(-beta.float()))).to(torch.bfloat16)   # [T, H]
    b = beta_active.float()
    vf = v.float()

    outputs, checkpoints, finals = [], [], []
    for s in range(sequences):
        state = initial_state[s].to(torch.bfloat16).float()                   # [H, V, K]
        for t in range(bounds[s], bounds[s + 1]):
            if (t - bounds[s]) % chunk == 0:
                checkpoints.append(state.to(torch.bfloat16))
            decayed = state * decay[t].unsqueeze(1)
            prediction = (decayed * kn[t].unsqueeze(1)).sum(dim=-1)          # [H, V]
            update = (vf[t] - prediction) * b[t].unsqueeze(-1)
            state = decayed + update.unsqueeze(-1) * kn[t].unsqueeze(1)
            outputs.append((state * qn[t].unsqueeze(1)).sum(dim=-1) * scale)
        finals.append(state)
    out = torch.stack(outputs, dim=0).to(torch.bfloat16)                      # [T, H, V]
    return (out, torch.stack(checkpoints, dim=0), beta_active,
            torch.stack(finals, dim=0))
