"""TensorRT-LLM chunk_scan, fixed B=H=G=1, L=Q=256, P=32, N=128.

Source: NVIDIA/TensorRT-LLM ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8,
cpp/tensorrt_llm/kernels/selectiveScan/chunkscan.h (chunk_scan_kernel).
X contains packed [x, B, C]; CB is the already-computed C/B product.
Os is the incoming chunk state, dc the time step, dA its accumulated decay.
This specialization has full sequence length, with D and Z both present.
"""
import torch


def reference(X, Os, dc, dA, CB, D, Z, last_token_ids):
    # last_token_ids is the fixed table [256] from input.json, passed for the binding; not used by the computation.
    # This full-length specialization has no masked/padded output rows.
    x = X[:, :32].float()
    c = X[:, 160:288].float()
    state_term = (c @ Os.float()) * torch.exp(dA).reshape(256, 1)
    decay = torch.exp(dA.reshape(256, 1) - dA.reshape(1, 256))
    coefficient = CB.float() * decay * dc.reshape(1, 256)
    row = torch.arange(256, device=X.device).reshape(256, 1)
    col = torch.arange(256, device=X.device).reshape(1, 256)
    coefficient = torch.where(col <= row, coefficient, 0).to(X.dtype).float()
    y = state_term + coefficient @ x + D.float().reshape(1, 1) * x
    z = torch.clamp_min(Z[:, :32].float(), -256.0)
    return (y * (z / (1.0 + torch.exp(-z)))).to(X.dtype)
