# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_sm100.py
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/interface.py
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/tile_scheduler.py
import torch
from examples._reference.pytorch_models import gemm, attention, mamba, kda


def reference(Q, K, V, scale_log2):
    return attention(Q, K, V, scale_log2, causal=False, splits=2, output_dtype=torch.float32)
