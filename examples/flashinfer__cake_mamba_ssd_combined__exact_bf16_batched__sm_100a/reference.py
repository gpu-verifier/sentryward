# Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/cake_mamba_ssd_combined/generated/device/sm_100a/mamba_ssd_q_tmem_alias_bf16_batched_152ad01e4f.cu
import torch
from examples._reference.pytorch_models import gemm, attention, mamba, kda


def reference(X, B, C, delta, cumsum):
    return mamba(X, B, C, delta, cumsum)
