"""FlashMLA SM100 fused Q-norm + Q-RoPE + sparse attention + O-RoPE + O-FP8-cast prefill kernel, h_q = 128 (2-CTA
cluster), V4.1 bf16 KV, enable_q_norm = true, no attention sink, no topk_length (out_fp8, out_sf, max_logits, lse).

Source (FlashMLA 395cb3afbb331fef7b1b21e9340861b9436b42d8; TileKernels c1cf914a82e6901c679445f872051ce44ac2cc63):
  csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/config.h:1-67 (the fused
    stages; q in the permuted layout of permute_q_b_proj; out_fp8 / out_sf in the layout of the permute_wv_proj weights),
  core_attn/kernel.cuh:553-659 (Q load: element (h, d) of a token at (d / 16) * (H_Q * 16) + h * 16 + d % 16; sum of
    squares of the bf16 Q before RoPE; non-neox RoPE of dims [448, 512) with cos = cache[pos, 0:32],
    sin = cache[pos, 32:64], rounded to bf16), :906-917 (score multiplier = sm_scale_div_log2 *
    rsqrt(sum_sq / 512 + eps)), :949-1016 (masked exp2 softmax, max_logits = real_mi * CUDART_LN2_F),
    :660-799 (lse = mi * ln2 + ln(li), -inf -> +inf; output_scale = 1 / li, 0 if li == 0; conjugate RoPE of the last 64
    dims; per 32-element tile: amax clamped to 1e-4, sf = amax / 448, exp = ceil(log2(sf)) from the f32 bits,
    fp8 = satfinite e4m3 of o * 2^-exp, sf byte = exp + 127; tile (head h = 8 g + hg, dims 32 t ..) at out_fp8 byte
    s * 65536 + g * 4096 + t * 256 + hg * 32 and at sf byte hg + 8 t of word row (s, g)),
    csrc/cuda_kernels/sm100/common_subroutine.h:74-223 (masking, P reduction, get_s_from_p);
  csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/kernel.cu:24-27 (q layout);
  csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp:15-28, :126-176 (outputs, sf allocation, parameters);
  tests/test-fused-norm-rope-attn-rope-cast.py:65-105 (ref_fused_norm_rope_attn_rope_cast_fwd: RMS factor from the q
    before RoPE, Q RoPE then bf16, ref_sparse_attn_fwd, conjugate O RoPE), :272-286 (out layout: view(s_q, n_wv_group,
    d_v / 32, wv_group_size, 32)), tests/ref.py:19-57 (indices < 0 or >= s_kv invalid; max_logits = max P; lonely
    rows: lse = +inf, O = 0);
  TileKernels tile_kernels/quant/common.py:251-282 (get_sf_and_inv with round_sf and packed ue8m0:
    exp_sf = ((bits(sf) - 1) >> 23) + 1 - 127, sf_inv = bits((127 - exp_sf) << 23), sf byte = exp_sf + 127),
    tile_kernels/config.py:10-11 (e4m3 amax clamp 1e-4).
The kernel's running maximum mi cancels in ideal-real arithmetic, as in the corpus attention references; scores use
the stored sm_scale_div_log2 scalar with exp2 as the kernel does, and max_logits uses the f32 value of CUDART_LN2_F.
The kernel applies the O RoPE and the cast to the unnormalized O times 1 / li; both are linear in O, so the reference
normalizes first. `x << 23` is written as `x * 2**23` (the reference lowering has no left shift).
KV and indices are kernelkit non_contiguousify views: the reference receives their whole contiguous storage tensors
and slices the logical blocks with the kernel scalars s_kv and topk (h_kv = 1).
"""
import torch

LN2 = 0.6931471824645996  # CUDART_LN2_F (math_constants.h: 6.931471805e-1F) as float32
D_QK = 512                # = d_v; V is the whole token
D_ROPE = 64               # RoPE dims [D_QK - D_ROPE, D_QK)
WV_GROUP_SIZE = 8
TILE = 32                 # O quantization tile (num_per_channels)


def _rope(x, cos, sin):
    # Non-neox RoPE of interleaved pairs (2i, 2i+1): (x0 cos - x1 sin, x0 sin + x1 cos) (kernel.cuh:26-33).
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).reshape(x.shape)


def reference(Q, KV, indices, token_positions, cos_sin_cache, sm_scale_log2, rms_norm_eps, s_kv, h_q, topk):
    # Q: bf16 [s_q, h_q, 512] buffer in the permuted layout.  KV: bf16 storage [s_kv+1, 2, 640], logical block
    # [s_kv, 1, 512].  indices: int32 storage [s_q+1, 2, topk+128], logical block [s_q, 1, topk].
    # token_positions: int32 [s_q].  cos_sin_cache: f32 [131072, 64].
    s_q = Q.shape[0]
    q = Q.reshape(s_q, D_QK // 16, h_q, 16).permute(0, 2, 1, 3).reshape(s_q, h_q, D_QK).float()

    # Q RMS-norm factor folded into the log2-unit score scale (sum of squares before RoPE)
    multiplier = sm_scale_log2 * torch.rsqrt((q * q).sum(dim=-1) / D_QK + rms_norm_eps)     # [s_q, h_q]

    table = cos_sin_cache[token_positions]                                   # [s_q, 64]
    cos = table[:, :D_ROPE // 2].unsqueeze(1)                                # [s_q, 1, 32]
    sin = table[:, D_ROPE // 2:].unsqueeze(1)
    q_rope = _rope(q[..., D_QK - D_ROPE:], cos, sin).to(torch.bfloat16).float()
    q = torch.cat((q[..., :D_QK - D_ROPE], q_rope), dim=-1)

    kv = KV[:s_kv, 0, :D_QK].float()                                         # [s_kv, 512]
    idx = indices[:s_q, 0, :topk]                                             # [s_q, topk]
    valid = (idx >= 0) & (idx < s_kv)
    rows = torch.where(valid, idx, 0).reshape(-1).long()
    gathered = kv.index_select(0, rows).reshape(s_q, topk, D_QK)             # [s_q, topk, 512]

    qk = q @ gathered.transpose(-1, -2)                                      # [s_q, h_q, topk]
    mask = valid.unsqueeze(1)
    max_logits = torch.where(mask, qk, float("-inf")).amax(dim=-1) * multiplier * LN2
    weight = torch.where(mask, torch.exp2(qk * multiplier.unsqueeze(-1)), 0.0)
    total = weight.sum(dim=-1)                                               # [s_q, h_q]
    lse = torch.where(total == 0.0, float("inf"), torch.log(total))
    scale = torch.where(total == 0.0, 0.0, torch.reciprocal(total))
    out = (weight @ gathered) * scale.unsqueeze(-1)                          # [s_q, h_q, 512]

    # Conjugate O RoPE of the last 64 dims
    out = torch.cat((out[..., :D_QK - D_ROPE], _rope(out[..., D_QK - D_ROPE:], cos, -sin)), dim=-1)

    # Per-32-element FP8 cast with a power-of-two (ue8m0) scale
    o = out.reshape(s_q, h_q, D_QK // TILE, TILE)                            # [s_q, h_q, 16, 32]
    sf = o.abs().amax(dim=-1).clamp_min(1e-4) / 448.0                        # [s_q, h_q, 16]
    exp_sf = ((sf.view(torch.int32) - 1) >> 23) - 126                        # ceil(log2(sf))
    sf_inv = ((127 - exp_sf) * 8388608).view(torch.float32)                  # 2^-exp_sf
    scaled = (o * sf_inv.unsqueeze(-1)).clamp(-448.0, 448.0)                 # satfinite

    n_group = h_q // WV_GROUP_SIZE
    # out_fp8 [s_q, n_wv_group, 8 * 512]: (g, t, hg, e) holds head 8 g + hg, dim 32 t + e
    out_fp8 = scaled.reshape(s_q, n_group, WV_GROUP_SIZE, D_QK // TILE, TILE).permute(0, 1, 3, 2, 4)
    out_fp8 = out_fp8.reshape(s_q, n_group, WV_GROUP_SIZE * D_QK).to(torch.float8_e4m3fn)
    # out_sf bytes [s_q, n_wv_group, 32 words, 4 bytes]: byte hg + 8 t of the row holds exp + 127
    out_sf = (exp_sf + 127).reshape(s_q, n_group, WV_GROUP_SIZE, D_QK // TILE).permute(0, 1, 3, 2)
    out_sf = out_sf.reshape(s_q, n_group, WV_GROUP_SIZE * D_QK // TILE // 4, 4).to(torch.uint8)
    return out_fp8, out_sf, max_logits, lse
