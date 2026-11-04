# Source: https://github.com/NVIDIA/TensorRT-LLM/blob/7587cc8e8a61a39be9d05ee00bf612088ee675c9/cpp/tensorrt_llm/kernels/kdaDecode/kdaDecodeLegacy.cu
# Source: https://github.com/NVIDIA/TensorRT-LLM/blob/7587cc8e8a61a39be9d05ee00bf612088ee675c9/cpp/tensorrt_llm/kernels/kdaDecode/kdaDecode.h
# Source: https://github.com/NVIDIA/TensorRT-LLM/blob/7587cc8e8a61a39be9d05ee00bf612088ee675c9/tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py
# Source: https://github.com/NVIDIA/TensorRT-LLM/blob/7587cc8e8a61a39be9d05ee00bf612088ee675c9/tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py
#
# Single-token KDA decode for H heads (K = V = 128, conv width 4) on the production fast path
# (KimiKDALinearAttention.forward_decode): an int32 slot table ssm_state_indices addresses an fp32 state pool and
# the conv-window pools, update_conv_cache = True (the window shifts in the new token), lower-bound gate, in-kernel
# beta sigmoid and fused output RMSNorm gate. The per-head math is the legacy kernel's and the FLA reference test's.
import torch


def reference(x_q, x_k, x_v, w_q, w_k, w_v, bias_q, bias_k, bias_v, cs_q, cs_k, cs_v, a_log, g, dt_bias, beta,
              onorm_g, onorm_weight, ssm_state_indices, state, lower_bound, scale, epsilon):
    # x_*, g, onorm_g: [B, H, 128]; w_*: [4, H*128] (transposed depthwise conv weights); bias_*, dt_bias: [H*128]
    # cs_*: [slots, H*128, 3] conv-window pools; a_log: [H]; beta: [B, H]; onorm_weight: [128]
    # ssm_state_indices: [B] int32 slots; state: [slots, H, 128 (v), 128 (k)] pool, updated in place by the kernel.
    B, H, D = x_q.shape
    slots = state.shape[0]
    if B != 1:
        raise ValueError("binding covers one batch row (one slot)")
    slot = ssm_state_indices[:B].to(torch.int64)
    rows = torch.arange(slots, device=state.device)
    hit = rows == slot[0]

    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))

    def convolution(x, w, bias, pool):
        cache = torch.index_select(pool, 0, slot)
        window = cache.float().reshape(B, H, D, 3)
        weight = w.float().reshape(4, H, D)
        value = (window * weight[:3].permute(1, 2, 0)).sum(dim=-1) + x.float() * weight[3] \
            + bias.float().reshape(H, D)
        shifted = torch.cat((cache[:, :, 1:], x.reshape(B, H * D, 1)), dim=-1)
        return value * sigmoid(value), torch.where(hit.reshape(slots, 1, 1), shifted, pool)

    q, new_cs_q = convolution(x_q, w_q, bias_q, cs_q)
    k, new_cs_k = convolution(x_k, w_k, bias_k, cs_k)
    v, new_cs_v = convolution(x_v, w_v, bias_v, cs_v)
    q = q * scale * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    decay = torch.exp(lower_bound * sigmoid(torch.exp(a_log).reshape(1, H, 1) * (g.float() + dt_bias.reshape(H, D))))
    decayed = torch.index_select(state, 0, slot) * decay.unsqueeze(-2)
    prediction = (decayed * k.unsqueeze(-2)).sum(dim=-1)
    update = (v - prediction) * sigmoid(beta.float()).unsqueeze(-1)
    next_state = decayed + update.unsqueeze(-1) * k.unsqueeze(-2)
    out = (next_state * q.unsqueeze(-2)).sum(dim=-1)
    out = out * torch.rsqrt((out * out).mean(dim=-1, keepdim=True) + epsilon)
    out = out * onorm_weight * sigmoid(onorm_g.float())
    new_state = torch.where(hit.reshape(slots, 1, 1, 1), next_state[0], state)
    return new_cs_q, new_cs_k, new_cs_v, new_state, out.to(x_q.dtype)
