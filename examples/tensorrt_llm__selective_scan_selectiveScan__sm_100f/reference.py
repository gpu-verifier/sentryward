"""TensorRT-LLM Mamba-1 selective scan (selective_scan_loop_kernel<float, float, 16, 128, 12, 6>).

Source: NVIDIA/TensorRT-LLM ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8,
cpp/tensorrt_llm/kernels/selectiveScan/selectiveScan.cu:79-323 (kernel),
cpp/tensorrt_llm/kernels/selectiveScan/selectiveScan.h:31-59 (SSMParamsBase).

Padded input (remove_padding=false): sample b owns rows b*max_seqlen ...,
and processes last_token_ids[b] tokens. The scan state starts at zero (the
kernel does not read the cache) and its final value is stored to cache slot
slot_mapping[b] (paged state; slot_mapping is bounded to [0, slots-1] by the
input's tensor_properties). BC packs [dt_rank columns (unused here), B (dstate),
C (dstate)] per token. Workspace pointers are null on this mamba-1 path.
"""
import torch


def reference(u, delta, A, BC, D, delta_bias, z, state, slot_mapping,
              last_token_ids, batch, dim, dstate, dt_rank, max_seqlen,
              last_token_id0, remove_padding, delta_softplus):
    # last_token_ids is the fixed shape-array table, passed for the binding;
    # its concrete entry enters as the scalar last_token_id0.
    if remove_padding:
        raise ValueError("This reference covers the padded layout only")
    if not delta_softplus:
        # The source leaves dt_b_sp uninitialized when softplus is disabled.
        raise ValueError("This reference covers delta_softplus=true only")
    if batch != 1:
        raise ValueError("last_token_id0 binds sample 0 only")
    tokens = last_token_id0
    if tokens < 1 or tokens > max_seqlen:
        raise ValueError("last_token_ids[0] must lie in [1, max_seqlen]")
    if last_token_ids.shape != (batch,):
        raise ValueError("last_token_ids must have one entry per sample")
    if u.shape != (batch, max_seqlen, dim) or A.shape != (dstate, dim):
        raise ValueError("unexpected tensor shapes")
    if BC.shape[-1] != dt_rank + 2 * dstate:
        raise ValueError("BC must pack dt_rank + 2*dstate columns")

    a = A.float()
    s = torch.zeros((batch, dstate, dim), dtype=torch.float32, device=u.device)
    outputs = []
    for t in range(tokens):
        dt_b = delta[:, t, :].float() + delta_bias.float()
        dt_sp = torch.where(dt_b <= 20.0, torch.log(1.0 + torch.exp(dt_b)), dt_b)
        x_t = u[:, t, :].float()
        b_t = BC[:, t, dt_rank:dt_rank + dstate].float()
        c_t = BC[:, t, dt_rank + dstate:dt_rank + 2 * dstate].float()
        decay = torch.exp(a.reshape(1, dstate, dim) * dt_sp.reshape(batch, 1, dim))
        s = s * decay + b_t.reshape(batch, dstate, 1) * (dt_sp * x_t).reshape(batch, 1, dim)
        y = (s * c_t.reshape(batch, dstate, 1)).sum(dim=1) + x_t * D.float()
        z_t = z[:, t, :].float()
        y = y * (z_t * torch.reciprocal(1.0 + torch.exp(-z_t)))
        outputs.append(y)
    out = torch.stack(outputs, dim=1).to(u.dtype)

    # Final state goes to cache slot slot_mapping[b]; other slots keep their
    # initial contents.
    slots = state.shape[0]
    slot_ids = torch.arange(slots, device=state.device)
    target = slot_mapping.to(torch.int64)
    result = state.clone()
    final = s.to(state.dtype)
    for sample in range(batch):
        hit = slot_ids == target[sample]
        result = torch.where(hit.reshape(slots, 1, 1), final[sample], result)
    return out, result
