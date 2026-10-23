"""CUTLASS example 112 Blackwell SSD (Mamba-2 chunked state-space scan), HAS_D with per-dim D, no Z.

Inputs use the example's cuDNN layouts: x [B, EH, D, C, L], delta [B, EH, C, L],
delta_a_cumsum [B, EH, C, L] (fp32 inclusive per-chunk cumsum of delta_A, produced by CumsumKernel),
b, c [B, G, N, C, L], d [EH, D]. Outputs: y as [B, EH, C, L, D] and the final state p as [B, EH, N, D].

Sources (CUTLASS e64a9136dd929639e5f7c969fe5af3bf7415cd4f, examples/112_blackwell_ssd/):
  reference/reference_ssd.hpp:73-112,169-312 (segsum; ssd_reference_impl: intra/inter BMMs, state recurrence, y, fstate)
  reference/reference_ssd_cumsum.hpp:138-188 (cumsum = inclusive per-chunk running sum, stored as float)
  collective/sm100_ssd_gemm_tma_warpspecialized.hpp:910-975 (exp(last - cs_j) * delta_j, state + exp(last) * state),
    :1097-1110 (segsum exp(cs_i - cs_j) * delta_j * (C_i . B_j), zero for j > i)
  collective/sm100_ssd_epilogue.hpp:411-416 (y = intra + inter * exp(cs_i) + D * x)
  112_blackwell_ssd.cu:66-90,160-305,544-568 (bf16 data, float cumsum, layouts, kernel arguments)
"""
import torch


def reference(x, delta, delta_a_cumsum, b, c, d, groups):
    batch, heads, dim, chunks, length = x.shape
    state_dim = b.shape[2]
    ratio = heads // groups
    group_of_head = torch.arange(heads, device=x.device) // ratio

    def per_head(t):  # [B, G, N, C, L] -> [B, EH, C, L, N]; head eh uses group eh // ratio
        return t.float().index_select(1, group_of_head).permute(0, 1, 3, 4, 2)

    B_, C_ = per_head(b), per_head(c)                       # [B, EH, C, L, N]
    X = x.float().permute(0, 1, 3, 4, 2)                     # [B, EH, C, L, D]
    dt = delta.float()                                       # [B, EH, C, L]
    cs = delta_a_cumsum.float()                              # [B, EH, C, L]

    # Intra-chunk: y_intra[i] = sum_{j <= i} exp(cs_i - cs_j) * dt_j * (C_i . B_j) * x_j
    i = torch.arange(length, device=x.device).reshape(length, 1)
    j = torch.arange(length, device=x.device).reshape(1, length)
    decay = torch.where(j <= i, torch.exp(cs.unsqueeze(-1) - cs.unsqueeze(-2)), 0.0)
    scores = C_ @ B_.transpose(-1, -2)                        # [B, EH, C, L(i), L(j)]
    y_intra = (decay * dt.unsqueeze(-2) * scores) @ X        # [B, EH, C, L, D]

    # Chunk states: contrib[c] = sum_j exp(last_c - cs_j) * dt_j * B_j^T x_j; S[0] = 0; S[c+1] = contrib[c] + exp(last_c) S[c]
    last = cs[..., length - 1]                                # [B, EH, C]
    weight = torch.exp(last.unsqueeze(-1) - cs) * dt          # [B, EH, C, L]
    contrib = (B_ * weight.unsqueeze(-1)).transpose(-1, -2) @ X   # [B, EH, C, N, D]
    state = torch.zeros((batch, heads, state_dim, dim), dtype=torch.float32, device=x.device)
    states = []
    for chunk in range(chunks):
        states.append(state)
        state = contrib[:, :, chunk] + torch.exp(last[:, :, chunk]).reshape(batch, heads, 1, 1) * state
    y_inter = C_ @ torch.stack(states, dim=2)                 # [B, EH, C, L, D]

    y = y_inter * torch.exp(cs).unsqueeze(-1) + y_intra + X * d.float().reshape(1, heads, 1, 1, dim)
    return y.to(torch.bfloat16), state.to(torch.bfloat16)
