"""Unoptimized mathematical references with explicit tensor arguments."""
import torch


def gemm(A, B):
    return (A.float() @ B.float().T).to(A.dtype)


def rowwise_gemm(A, B, row_scale, column_scale):
    return ((A.float() @ B.float().T) * row_scale[:, None] * column_scale[None, :]).to(torch.bfloat16)


def rope_halves(x, cosine, sine):
    half = x.shape[-1] // 2
    first, second = x[..., :half].float(), x[..., half:].float()
    return torch.cat((first * cosine - second * sine,
                      second * cosine + first * sine), dim=-1).to(x.dtype)


def rope_positions(q, k, pos_ids, reciprocal_theta, reciprocal_scale):
    half = q.shape[-1] // 2
    frequency = reciprocal_theta ** (torch.arange(half, device=q.device).float() / half)
    angle = pos_ids.float()[:, None, None] * frequency * reciprocal_scale
    return rope_halves(q, angle.cos(), angle.sin()), rope_halves(k, angle.cos(), angle.sin())


def rope_cache(query, key, positions, cos_sin_cache):
    half = query.shape[-1] // 2
    cache = cos_sin_cache[positions].float().unsqueeze(1)
    return (rope_halves(query, cache[..., :half], cache[..., half:]),
            rope_halves(key, cache[..., :half], cache[..., half:]))


def fp8_per_token(input):
    # vLLM csrc/quantization/utils.cuh: min_scaling_factor<Float8_e4m3fn>.
    scale = (input.float().abs().amax(dim=-1) / 448).clamp_min(1 / (448 * 512))
    output = (input.float() / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return output, scale


def fused_qk_rope(qkv, q_weight, k_weight, position_ids, cos_sin_cache, epsilon):
    q, k = qkv[:, :3].float(), qkv[:, 3:4].float()
    q = q * torch.rsqrt((q * q).mean(dim=-1, keepdim=True) + epsilon) * q_weight.float()
    k = k * torch.rsqrt((k * k).mean(dim=-1, keepdim=True) + epsilon) * k_weight.float()
    q, k = rope_cache(q, k, position_ids, cos_sin_cache)
    return torch.cat((q, k), dim=1).to(qkv.dtype)


def attention_backward(Q, K, V, dO, LSE, dPsum, dQaccum, scale, scale_log2):
    probability = torch.exp2((Q.float() @ K.float().T) * scale_log2 - LSE[:, None])
    derivative = probability * ((dO.float() @ V.float().T) - dPsum[:, None])
    p = probability.to(Q.dtype).float()
    ds = derivative.to(Q.dtype).float()
    # dQ scaling is performed by the separate postprocess kernel.
    dq = dQaccum + ds @ K.float()
    dk = ((ds.T @ Q.float()) * scale).to(Q.dtype)
    dv = (p.T @ dO.float()).to(Q.dtype)
    return dq, dk, dv


def softmax(input):
    weight = torch.exp(input.float())
    return (weight / weight.sum(dim=-1, keepdim=True)).to(input.dtype)


def weight_norm(v, g):
    norm = torch.sqrt((v.float() * v.float()).sum(dim=-1))
    return (v.float() * (g.float() / norm).unsqueeze(-1)).to(v.dtype), norm


def batch_statistics(input, epsilon):
    x = input.float()
    mean = x.mean(dim=(0, 2))
    delta = x - mean.reshape(1, -1, 1)
    variance = (delta * delta).mean(dim=(0, 2))
    return mean, torch.rsqrt(variance + epsilon)


def layer_norm_backward(dY, X, mean, rstd, gamma):
    normalized = (X.float() - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)
    grad = dY.float() * gamma.float()
    centered = grad - grad.mean(dim=-1, keepdim=True)
    correction = normalized * (grad * normalized).mean(dim=-1, keepdim=True)
    return ((centered - correction) * rstd.unsqueeze(-1)).to(X.dtype)


def margin_backward(input, target, gradOutput, margin):
    selected = input.gather(1, target.reshape(-1, 1))
    classes = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    active = ((margin - selected + input) > 0) & (classes != target.reshape(-1, 1))
    positive = active.to(input.dtype) / input.numel()
    negative = positive.sum(dim=-1, keepdim=True)
    return torch.where(classes == target.reshape(-1, 1), -negative, positive) * gradOutput


def fused_add_rms(input, residual, weight, epsilon):
    summed = input.float() + residual.float()
    normalized = summed * torch.rsqrt((summed * summed).mean(dim=-1, keepdim=True) + epsilon)
    return (normalized * weight.float()).to(input.dtype), summed.to(residual.dtype)


def attention(Q, K, V, scale_log2, *, causal=False, splits=1, output_dtype=None):
    batch, heads, rows, dim = Q.shape
    kv_heads, columns = K.shape[1:3]
    group = heads // kv_heads
    q = Q.float().reshape(batch, kv_heads, group, rows, dim)
    outputs, lses = [], []
    for split in range(splits):
        begin, end = split * columns // splits, (split + 1) * columns // splits
        k = K[:, :, begin:end, :].float().unsqueeze(2)
        v = V[:, :, begin:end, :].float().unsqueeze(2)
        score = (q @ k.transpose(-1, -2)) * scale_log2
        if causal:
            query = torch.arange(rows, device=Q.device).reshape(rows, 1)
            key = torch.arange(begin, end, device=Q.device).reshape(1, end - begin)
            score = torch.where(key <= query + columns - rows, score, -float("inf"))
        # Common row scaling cancels under ideal-real semantics.
        weight = torch.exp2(score)
        total = weight.sum(dim=-1, keepdim=True)
        outputs.append(((weight @ v) / total).reshape(batch, heads, rows, V.shape[-1]))
        lses.append(torch.log(total).reshape(batch, heads, rows))
    if splits == 1:
        return outputs[0].to(output_dtype or Q.dtype)
    return torch.stack(outputs), torch.stack(lses)


def mamba(X, B, C, delta, cumsum):
    x, b, c = X.float(), B.float(), C.float()
    dt = delta.float().reshape(1, 256, 1)
    cs = cumsum.reshape(1, 256, 1)
    state = torch.zeros((1, 1, 64, 128), dtype=torch.float32, device=X.device)
    outputs = []
    for t in range(256):
        previous = 0 if t % 128 == 0 else cs[:, t - 1]
        decay = torch.exp(cs[:, t] - previous).unsqueeze(-1).unsqueeze(-1)
        update = (dt[:, t].unsqueeze(-1) * x[:, t]).unsqueeze(-1) * b[:, t].unsqueeze(-2)
        state = decay * state + update
        outputs.append((state * c[:, t].unsqueeze(-2)).sum(dim=-1))
    return torch.stack(outputs, dim=1).to(X.dtype), state.to(X.dtype)


def kda(x_q, x_k, x_v, w_q, w_k, w_v, bias_q, bias_k, bias_v,
        cs_q, cs_k, cs_v, a_log, g, dt_bias, beta, onorm_g, onorm_weight,
        state, lower_bound, scale, epsilon):
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))
    def convolution(x, w, bias, cache):
        value = (cache.float() * w[:3].float().T).sum(dim=-1) + x.float() * w[3].float() + bias.float()
        return value * sigmoid(value), torch.cat((cache[:, :, 1:], x.unsqueeze(-1)), dim=-1)
    q, cq = convolution(x_q, w_q, bias_q, cs_q)
    k, ck = convolution(x_k, w_k, bias_k, cs_k)
    v, cv = convolution(x_v, w_v, bias_v, cs_v)
    q = q * scale * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    decay = torch.exp(lower_bound * sigmoid(torch.exp(a_log) * (g.float() + dt_bias)))
    decayed = state * decay.unsqueeze(-2)
    prediction = (decayed * k.unsqueeze(-2)).sum(dim=-1)
    update = (v - prediction) * sigmoid(beta.float()).unsqueeze(-1)
    next_state = decayed + update.unsqueeze(-1) * k.unsqueeze(-2)
    out = (next_state * q.unsqueeze(-2)).sum(dim=-1)
    out = out * torch.rsqrt((out * out).mean(dim=-1, keepdim=True) + epsilon)
    out = out * onorm_weight * sigmoid(onorm_g.float())
    return cq, ck, cv, next_state, out.to(x_q.dtype)
