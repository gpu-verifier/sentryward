# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/interface.py
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/tile_scheduler.py
import torch
from examples._reference.pytorch_models import gemm, attention, mamba, kda


def reference(Q, K, V, scale_log2):
    return attention(Q, K, V, scale_log2, causal=False, splits=1, output_dtype=torch.bfloat16)
