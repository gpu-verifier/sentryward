"""TIRx KDA forward, fused packed-varlen route (curated_kda_forward_portfolio_multishape, h96_fixed).

Source: https://github.com/mlc-ai/TIRx-kernels/blob/ff128be7470d674a94169e12c1556590e4046cba/tirx_kernels/kda/kda_forward_portfolio_multishape.py
(module docstring L4-17; fused_setup L1966-2030; CONFIGS h96_fixed L4004-4014 and KDAForwardPortfolioConfig L3971-3977:
scale 1/sqrt(128), lower_bound -5; inputs _make_case L4102-4140; oracle _reference L4179-4202). The recorded revision
532949266f is not public; its kernel ABI differs from ff128be7 (no item tables), see the binding provenance.
Oracle: fla.ops.kda.chunk_kda at flash-linear-attention 9c8e42e762fce087c27b673af4922795d9edb85e with
use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, safe_gate, lower_bound, state_v_first,
initial_state and output_final_state (https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/chunk.py
L178-L299). Its upstream reference implementations are fla/ops/kda/naive.py (naive_recurrent_kda L12-L65 and
naive_chunk_kda L68-L155), fla/ops/kda/gate.py (naive_kda_lowerbound_gate L58-L69) and fla/modules/l2norm.py (eps 1e-6).

Per head h and token t of the one sequence (cu_seqlens [0, T]):
  q' = scale * q * rsqrt(sum q^2 + 1e-6), k' = k * rsqrt(sum k^2 + 1e-6)
  g' = lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias[h])) (log-space decay), b = sigmoid(beta)
  S = S * exp(g') ; S = S + b k' (v - k'^T S)^T ; o_t = S^T q'   (S is [K, V]; the state is stored V-first)
The state starts from h0 and final_state = S after the last token.

Form: naive_chunk_kda's evaluation with one chunk spanning the whole sequence; in real arithmetic it equals the
token recurrence for every chunk size. G is the cumulative log decay (a lower-triangular sum), and the forward
substitution (I - L)^-1 of the strictly lower-triangular T x T matrix L is the finite product
(I + L)(I + L^2)(I + L^4)...(I + L^(2^(F-1))) with 2^F >= T, exact because L^T = 0. The reference capture builds
one expression per (node, coordinates, fold depth), and each step of a chunk recurrence nests one more fold, so a
chunk loop costs a number of expressions quadratic in the chunk count; one chunk keeps the capture linear.
"""
import torch

EPS = 1e-6


def _sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def reference(q, k, v, g, beta, A_log, dt_bias, h0, cu_start, cu_end, scale, lower_bound):
    _, total, heads, dim = q.shape
    if cu_start != 0 or cu_end != total or h0.shape[0] != 1:
        raise ValueError("binding covers one sequence")
    factors = max(1, (total - 1).bit_length())                                  # 2^factors >= total
    device = q.device

    qf, kf = q[0].float(), k[0].float()                                        # [T, H, K]
    qn = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + EPS) * scale
    kn = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + EPS)
    rate = torch.exp(A_log.float()).reshape(heads, 1)
    gate = lower_bound * _sigmoid(rate * (g[0].float() + dt_bias.float().reshape(heads, dim)))
    b = _sigmoid(beta[0].float()).transpose(0, 1)                               # [H, T]

    Q, K, V, gate = (x.permute(1, 0, 2) for x in (qn, kn, v[0].float(), gate))  # [H, T, D]
    row = torch.arange(total, device=device).reshape(total, 1)
    col = torch.arange(total, device=device).reshape(1, total)
    inclusive = col <= row
    strict = col < row
    G = torch.where(inclusive, 1.0, 0.0) @ gate                                 # cumulative log decay [H, T, K]

    # gap[h, c, i, d] = G[c, d] - G[i, d] where i <= c (0 elsewhere): exp(gap) <= 1 on the used entries.
    gap = torch.where(inclusive.unsqueeze(-1), G.unsqueeze(-2) - G.unsqueeze(-3), 0.0)
    decay = torch.exp(gap)
    kk = (K.unsqueeze(-2) * decay * K.unsqueeze(-3)).sum(dim=-1)                # [H, T, T]
    qk = (Q.unsqueeze(-2) * decay * K.unsqueeze(-3)).sum(dim=-1)
    lower = torch.where(strict, -b.unsqueeze(-1) * kk, 0.0)
    eye = torch.where(row == col, 1.0, 0.0)
    inverse = eye + lower
    power = lower
    for _ in range(factors - 1):
        power = power @ power
        inverse = inverse @ (eye + power)
    A = inverse * b.unsqueeze(-2)
    W = A @ (torch.exp(G) * K)
    U = A @ V
    Aqk = torch.where(inclusive, qk, 0.0)
    last = G[:, total - 1:total, :]                                             # [H, 1, K]

    state = h0[0].float().transpose(-1, -2)                                     # [H, K, V]
    corrected = U - W @ state                                                   # [H, T, V]
    out = (Q * torch.exp(G)) @ state + Aqk @ corrected
    carry = torch.exp(last - G) * K                                             # [H, T, K]
    final = state * torch.exp(last).transpose(-1, -2) + carry.transpose(-1, -2) @ corrected
    out = out.permute(1, 0, 2).reshape(1, total, heads, dim)
    final_state = final.transpose(-1, -2).reshape(1, heads, dim, dim)
    return out.to(q.dtype), final_state
