"""vLLM Kimi K3 fused KDA decode: short conv + KDA recurrent step + gated output RMSNorm.

Source: https://github.com/vllm-project/vllm/blob/605c3ddcbaaee06a2687a961ae7be51aec5dddc0/csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu
  lines 264-735 (kda_decode_fusion_many_heads_kernel), 737-782 (launcher, grid (B, 12)),
  816-825 (DS conv-state layout: channel stride 3, tap stride 1), 977-1162 (host op shapes).
Instantiation: <kApplyOnorm=1, kUseStaticDecodeLayout=1, kFixedHeads=12, kFixedValueHeads=12,
  kUseHeadGrid=1, kAccumulateOnormSumsq=0, kUseActiveQkReduction=0, kUseCacheGlobalStore=0,
  kComputeOutputBeforeStore=0, kSkipWarpSync=0, kPreloadOnormParams=1, kPrefetchNextStateChunk=1,
  kUseActiveOnormReduction=1, kUpdateConvState=1, kUseLowerBound=1, kApplyBetaSigmoid=1,
  kConvStateChannelStride=3, kConvStateTapStride=1>.

Per batch row n (slot = state_indices[n], no sentinel in the static layout) and head h:
  x' = silu(bias + sum_{t<3} cs[slot, c, t] * w[t, c] + x * w[3, c]); cs[slot, c] <- (cs[1], cs[2], x)
  q' = q * scale * rsqrt(sum q^2 + 1e-6), k' = k * rsqrt(sum k^2 + 1e-6)
  decay = exp(lower_bound * sigmoid(exp(a_log[h]) * (g + dt_bias)))
  S' = S * decay ; u = (v - S' k') * sigmoid(beta) ; S_new = S' + u k'^T (fp32 state)
  o = S_new q' ; out = o * rsqrt(mean(o^2) + eps) * onorm_weight * sigmoid(onorm_g)
Unselected slots of the conv caches and the state pool are returned unchanged.
"""
import torch


def _sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def reference(x_q, x_k, x_v, w_q, w_k, w_v, bias_q, bias_k, bias_v, cs_q, cs_k, cs_v,
              a_log, g, dt_bias, beta, onorm_g, onorm_weight, state_indices, state,
              batch, heads, value_heads, lower_bound, scale, onorm_eps):
    slots, num_heads, dim_v, dim_k = state.shape
    if batch != 1 or heads != value_heads or heads != num_heads or state_indices.shape[0] != batch:
        raise ValueError("binding covers batch 1 with H = HV = 12 (static decode layout)")
    slot = state_indices.to(torch.int64)                                   # [B]
    selected = torch.arange(slots, device=state.device) == slot[0]         # [S]

    def convolution(x, w, bias, cache):
        # x [B, H, D] bf16; w [4, H*D] f32 (tap-major); bias [H, D] f32; cache [S, H, D, 3] bf16
        history = cache.index_select(0, slot)                              # [B, H, D, 3]
        taps = w.float().reshape(4, num_heads, dim_k)
        acc = bias.float() + x.float() * taps[3]
        for t in range(3):
            acc = acc + history[..., t].float() * taps[t]
        shifted = torch.cat((history[..., 1:], x.unsqueeze(-1)), dim=-1)   # [B, H, D, 3]
        new_cache = torch.where(selected.reshape(slots, 1, 1, 1), shifted[0].unsqueeze(0), cache)
        return acc * _sigmoid(acc), new_cache

    q, new_cs_q = convolution(x_q, w_q, bias_q, cs_q)
    k, new_cs_k = convolution(x_k, w_k, bias_k, cs_k)
    v, new_cs_v = convolution(x_v, w_v, bias_v, cs_v)
    q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    decay = torch.exp(lower_bound * _sigmoid(torch.exp(a_log.float()).unsqueeze(-1)
                                             * (g.float() + dt_bias.float())))   # [B, H, K]
    b = _sigmoid(beta.float())                                             # [B, H]

    initial = state.index_select(0, slot)                                  # [B, H, V, K] f32
    decayed = initial * decay.unsqueeze(2)
    prediction = (decayed * k.unsqueeze(2)).sum(dim=-1)                    # [B, H, V]
    update = (v - prediction) * b.unsqueeze(-1)
    next_state = decayed + update.unsqueeze(-1) * k.unsqueeze(2)           # [B, H, V, K]
    o = (next_state * q.unsqueeze(2)).sum(dim=-1)                          # [B, H, V]
    o = o * torch.rsqrt((o * o).mean(dim=-1, keepdim=True) + onorm_eps)
    out = (o * onorm_weight.float() * _sigmoid(onorm_g.float())).to(torch.bfloat16)
    new_state = torch.where(selected.reshape(slots, 1, 1, 1), next_state[0].unsqueeze(0), state)
    return new_cs_q, new_cs_k, new_cs_v, new_state, out
