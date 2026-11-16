"""FlashMLA SM100 head64 sparse FP8 decode, V3.2 KV-cache format, no-split path (O and LSE).

Source (FlashMLA 15f13e5030374295491c5ce31b02d7e63a7772c6):
  csrc/sm100/decode/head64/config.h:33-41 (D_Q=576, D_NOPE=512, D_ROPE=64, D_V=512, 128-wide quant tiles,
    656-byte tokens: 512 e4m3 NoPE | 4 f32 scales | 64 bf16 RoPE);
  csrc/sm100/decode/head64/kernel.cuh:683-711 (index -> token, -1 = invalid),
    :781-806 (NoPE dequant e4m3 * scale), :210-260 (masked exp2 softmax with sm_scale_div_log2),
    :301-325 (empty row: li = 0, LSE = +inf; no-split LSE = mi*ln2 + ln(li)), :335-385 (O = acc / li, bf16);
  tests/ref.py:55-103 (ref_sparse_attn_decode), tests/quant.py:81-104 (dequantize_k_cache, V32 layout).
Each NoPE tile is the e4m3 value times its f32 scale factor, used as an f32 value
(flash_mla/flash_mla_interface.py:97, tests/quant.py:97, 101-104). The kernel reads only the exponent field
of the f32 word (cvt.rz.ue8m0x2.f32, kernel.cuh:708-709). The two agree only on power-of-two scales, which
are the values upstream's quantizer produces (tests/quant.py:17-18, 44-46); that is an input premise, not
part of this reference. Exponent field 255 (an inf or NaN scale) needs no special case: the f32 value
propagates as in upstream's dequantization.
"""
import torch


def reference(Q, KV_nope, KV_scale, KV_rope, indices, scheduler_metadata, num_splits, sm_scale_log2):
    # scheduler_metadata [num_sm_parts, 8] / num_splits [b + 1]: fixed shape-array tensors (no-split schedule),
    # passed for the binding; not used by the computation.
    # Q: [b, s_q, h_q, 576] bf16.  KV_*: three typed views of the same packed cache bytes
    # [num_blocks, page_block_size, 656]: e4m3 bytes, f32 words (164/token), bf16 halves (328/token).
    # indices: [b, s_q, topk] int32 token ids into the flattened cache (-1 = invalid).
    b, s_q, h_q, d_qk = Q.shape
    num_blocks, block_size, token_bytes = KV_nope.shape
    tokens = num_blocks * block_size
    topk = indices.shape[-1]
    nope = KV_nope.reshape(tokens, token_bytes)[:, :512].float()
    scale = KV_scale.reshape(tokens, token_bytes // 4)[:, 128:132].float()  # f32 scale factor per 128-wide tile
    nope = (nope.reshape(tokens, 4, 128) * scale.unsqueeze(-1)).reshape(tokens, 512)
    rope = KV_rope.reshape(tokens, token_bytes // 2)[:, 264:328].float()
    key = torch.cat((nope, rope), dim=-1)  # [tokens, 576]; V = key[:, :512]
    valid = indices != -1
    rows = torch.where(valid, indices, 0).reshape(-1).long()
    gathered = key.index_select(0, rows).reshape(b, s_q, topk, d_qk)
    score = (Q.float() @ gathered.transpose(-1, -2)) * sm_scale_log2  # [b, s_q, h_q, topk]
    weight = torch.where(valid.unsqueeze(2), torch.exp2(score), 0.0)
    total = weight.sum(dim=-1, keepdim=True)
    empty = total == 0
    out = (weight @ gathered[..., :512]) * torch.where(empty, 0.0, torch.reciprocal(total))
    lse = torch.where(empty, float("inf"), torch.log(total)).reshape(b, s_q, h_q)
    return out.to(torch.bfloat16), lse
