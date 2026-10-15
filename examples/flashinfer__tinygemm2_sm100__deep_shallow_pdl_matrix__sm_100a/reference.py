"""Reference for FlashInfer tinygemm2_sm100.cu, revision b250399302acb2c4925b03416188c6678b762fbd."""
import torch


def reference(A, B, bias):
    return (A.float() @ B.float().T + bias.float()).to(torch.bfloat16)
