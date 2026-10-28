"""vLLM fused GDN decode, post-conv MTP kernel <float, 1, true>.

Source: https://github.com/vllm-project/vllm/blob/605c3ddcbaaee06a2687a961ae7be51aec5dddc0/csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu
  gdn_decode_post_conv_mtp_kernel: lines 150-374 (math), helpers 105-126,
  host shape/stride checks and launch: lines 376-597.
Template instance (PTX entry name): StateT=float, ValueHeadsPerKeyHead=1,
SigmoidGate=true. One request (grid.x), HV value heads (grid.y).

Gated delta rule per value head, token t (state h[v, k], v = value row,
k = key column, layout [slot, HV, 128 (v), 128 (k)]):
  q = q_raw * rsqrt(sum q_raw^2 + 1e-6) * scale;  k = k_raw * rsqrt(sum k_raw^2 + 1e-6)
  g = -exp(A_log) * softplus(a + dt_bias);  decay = exp(g);  beta = sigmoid(b)
  h = h * decay;  delta = (v - h k) * beta;  h = h + delta k^T;  o = h q
  out = bf16(o) * rsqrt(mean(bf16(o)^2) + eps) * norm_weight * sigmoid(gate)
The source slot is state_indices[req, accepted - 1]; slot <= 0 (or an
invalid accepted count) zero-fills out and skips the state.  After token t
the state is stored to slot state_indices[req * width + t] when that slot
is > 0.  The state is read-modify-write; its initial table is an input.
"""
import torch

DIM = 128          # kDimK = kDimV = 128 (lines 85-86)
MAX_MTP_TOKENS = 8  # kMaxMtpTokens (line 92)


def _sigmoid(x):
    # sigmoid_fast (lines 105-107)
    return 1.0 / (1.0 + torch.exp(-x))


def _softplus(x):
    # softplus_fast: x > 20 ? x : log1pf(expf(x)) (lines 113-115)
    return torch.where(x > 20.0, x, torch.log(1.0 + torch.exp(x)))


def reference(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens,
              num_accepted_tokens, state, output_gate, norm_weight,
              H, HV, state_indices_width, dt_bias_type, norm_weight_is_bf16,
              scale, norm_eps, bos, eos):
    # Static scope checks (scalar kernel arguments / fixed shape arrays).
    if dt_bias_type != 0 or norm_weight_is_bf16:
        raise ValueError("binding covers float32 dt_bias and float32 norm_weight")
    if HV != H:
        raise ValueError("template ValueHeadsPerKeyHead=1 requires HV == H")
    requests, width = state_indices.shape
    if requests != 1 or width != state_indices_width:
        raise ValueError("binding covers one request with state_indices [1, width]")
    tokens = mixed_qkv.shape[0]
    num_tokens = eos - bos
    if bos != 0 or eos != tokens or not 0 < num_tokens <= MAX_MTP_TOKENS:
        raise ValueError("binding covers one request spanning all L <= 8 rows")
    if mixed_qkv.shape[1] != 2 * H * DIM + HV * DIM:
        raise ValueError("mixed_qkv width must be 2*H*128 + HV*128")
    if num_tokens > width * requests:
        raise ValueError("destination slot index would leave state_indices")
    slots = state.shape[0]
    if slots < 2:
        raise ValueError("state table must contain slot 0 and at least one slot > 0")

    # Source slot (lines 174-179); invalid accepted counts select slot 0.
    accepted = num_accepted_tokens[0:1]
    accepted_ok = (accepted > 0) & (accepted <= width)
    column = torch.where(accepted_ok, accepted - 1, 0).to(torch.int64).reshape(1, 1)
    source = torch.gather(state_indices[0:1, :], 1, column).reshape(1)
    active = accepted_ok & (source > 0)

    initial = torch.index_select(state, 0, source)[0].float()  # [HV, V, K]
    h = [initial[head] for head in range(HV)]
    flat_indices = state_indices.reshape(-1)
    slot_ids = torch.arange(1, slots, dtype=torch.int32, device=state.device)
    table = state
    out_rows = []
    for t in range(num_tokens):
        row_index = cu_seqlens[0:1] + t  # token = bos + t (cu_seqlens[request])
        row = torch.index_select(mixed_qkv, 0, row_index)[0].float()
        a_row = torch.index_select(a, 0, row_index)[0].float()
        b_row = torch.index_select(b, 0, row_index)[0].float()
        gate_row = torch.index_select(output_gate, 0, row_index)[0].float()
        heads = []
        for head in range(HV):
            key_head = head  # value_head / ValueHeadsPerKeyHead (line 190)
            q_raw = row[key_head * DIM:(key_head + 1) * DIM]
            k_raw = row[H * DIM + key_head * DIM:H * DIM + (key_head + 1) * DIM]
            v = row[2 * H * DIM + head * DIM:2 * H * DIM + (head + 1) * DIM]
            q = q_raw * torch.rsqrt((q_raw * q_raw).sum(dim=-1, keepdim=True) + 1.0e-6) * scale
            k = k_raw * torch.rsqrt((k_raw * k_raw).sum(dim=-1, keepdim=True) + 1.0e-6)
            g = -torch.exp(A_log[head:head + 1].float()) * _softplus(
                a_row[head:head + 1] + dt_bias[head:head + 1].float())
            decay = torch.exp(g)
            beta = _sigmoid(b_row[head:head + 1])
            state_v = h[head] * decay                                   # lines 288
            hk = (state_v * k.reshape(1, DIM)).sum(dim=-1)              # lines 289-295
            delta = (v - hk) * beta                                     # lines 301-303
            state_v = state_v + delta.reshape(DIM, 1) * k.reshape(1, DIM)  # line 306
            h[head] = state_v
            o = (state_v * q.reshape(1, DIM)).sum(dim=-1)               # lines 307-311
            o = o.to(torch.bfloat16).float()  # shared_out is bf16 (lines 195, 313-320)
            rstd = torch.rsqrt((o * o).sum(dim=-1, keepdim=True) / DIM + norm_eps)
            y = o * rstd * norm_weight.float() * _sigmoid(gate_row[head])
            heads.append(torch.where(active, y, 0.0))  # zero path (lines 179-187)
        out_rows.append(torch.stack(heads, dim=0))
        # State store after token t (lines 323-336). Slot 0 is never written
        # (destination_slot > 0), so only rows 1..slots-1 can change.
        destination = flat_indices[t:t + 1]
        write = (slot_ids == destination) & active
        updated = torch.stack(h, dim=0).reshape(1, HV, DIM, DIM)
        rest = torch.where(write.reshape(slots - 1, 1, 1, 1), updated, table[1:])
        table = torch.cat((table[0:1], rest), dim=0)
    out = torch.stack(out_rows, dim=0).to(torch.bfloat16)
    return out, table
