# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/xqa/mha.cu
#   kernel_mha_impl 1553-2700 (scales 1671-1672, qkScale 1816-1818, fp8 K->bf16 conversion 1095-1125,
#   softmax 2117-2122, voScale and sinks 2614-2634)
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/xqa/mhaUtils.cuh
#   HeadPtr 67-88, KVCacheList 244-255, getPage 287-302
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/flashinfer/xqa.py
#   q_scale/kv_scale float defaults 200-201 -> null scale pointers 129-130, 143-144
# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/tests/attention/test_xqa.py
#   ref_attention 34-120
import torch


def reference(q, k_cache, v_cache, page_list, q_scale, kv_scale, attention_sinks, seq_len, seqLenList):
    # seqLenList: fixed shape-array tensor(s) bound only so the native fixed-input context finds a
    # common identity (fixed_input_context.hpp decode_verified); values are not used here.
    # q/output: [batch, beam=1, nbQHeads, 128] bf16; K/V pool: [pages, tokensPerPage, nbKHeads, 128] e4m3.
    batch, beam, q_heads, dim = q.shape
    pages_per_seq = page_list.shape[1]
    _, tokens_per_page, kv_heads, _ = k_cache.shape
    group = q_heads // kv_heads
    # tensor_properties keep every page index inside the pool, so HeadPtr's sign-bit null-page
    # path (mhaUtils.cuh 83-86) is outside this input's domain.
    pages = page_list.reshape(-1).long()

    def gather(cache):
        # Decode e4m3 explicitly; the kernel's e4m3 -> bf16 conversion is exact.
        selected = cache.float().index_select(0, pages)
        selected = selected.reshape(batch, pages_per_seq * tokens_per_page, kv_heads, dim)
        return selected.permute(0, 2, 1, 3)[:, :, :seq_len, :]

    k, v = gather(k_cache), gather(v_cache)
    query = q.float().reshape(batch, kv_heads, group, dim)
    score = query @ k.transpose(-1, -2)
    # qScalePtr and kvScalePtr are null: qkScale = qScale * kvCacheScale * rsqrt(HEAD_ELEMS), and the
    # normalized output is multiplied by voScale = kvCacheScale.
    # Row-max subtraction cancels under ideal-real semantics, including the sink term exp(sink - max).
    weight = torch.exp(score * (q_scale * kv_scale) * torch.rsqrt(torch.ones_like(score) * dim))
    total = weight.sum(dim=-1, keepdim=True) + torch.exp(attention_sinks).reshape(1, kv_heads, group, 1)
    out = (weight @ v) * kv_scale / total
    return out.reshape(batch, beam, q_heads, dim).to(q.dtype)
