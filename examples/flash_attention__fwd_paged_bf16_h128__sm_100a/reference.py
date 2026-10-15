# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_sm100.py
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/paged_kv.py
import torch
from examples._reference.pytorch_models import attention


def reference(Q, K, V, page_table, scale_log2):
    batch, pages_per_sequence = page_table.shape
    _, page_size, heads, dim = K.shape
    pages = page_table.reshape(-1).long()
    def gather(cache):
        selected = cache.index_select(0, pages)
        selected = selected.reshape(batch, pages_per_sequence, page_size, heads, dim)
        return selected.permute(0, 3, 1, 2, 4).reshape(batch, heads, pages_per_sequence * page_size, dim)
    return attention(Q, gather(K), gather(V), scale_log2, output_dtype=torch.bfloat16)
