"""CAKE mamba SSD combined, exact bf16 varlen device kernel.

Source: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/csrc/cake_mamba_ssd_combined/generated/device/sm_100a/mamba_ssd_q_tmem_alias_bf16_varlen_351b79a64d.cu
  kernel_mamba_ssd_q_tmem_alias_bf16_varlen, lines 430-1610.  Work item
  (sequence, head) = (work / nheads, work % nheads); logical chunks
  [seq_chunk_cumsum[s], seq_chunk_cumsum[s+1]) (has_seq_chunk_cumsum, e.g.
  1107-1110); logical chunk l covers physical chunk chunk_indices[l], rows
  [chunk_offsets[l], limit) with limit = chunk_offsets[l+1] when chunk l+1 is
  in the same physical chunk, else 128 (1133-1144).  Factor loads
  (cumsum/delta_precomputed[(l*nheads+h)*128 + row - offset], 1041-1055);
  initial state (1082-1102); per-chunk state update (1183-1280); final_states
  (1308-1325); causal intra term (1391-1411); output with inter term
  exp(cs[r]) * C.state (1514-1578).
Host shapes and routing: https://github.com/flashinfer-ai/flashinfer/blob/b250399302acb2c4925b03416188c6678b762fbd/flashinfer/mamba/cake_ssd_combined.py

The math equals the batched sibling (same generator; only routing differs).
Per (sequence, head), over the sequence's segments in order, with the
segment-local precomputed cumsum cs and delta (segment base 0):
  state = exp(cs[t] - cs[t-1]) * state + delta[t] * x[t] b[t]^T   (cs[off-1] := 0)
  y[t]  = state . c[t]
starting from initial_states[sequence, head]; final_states gets the state after
the last segment.  dt, A and dt_bias are consumed by the preprocess kernel.
Intermediate BF16 rounding of chunk operands is abstracted as in the sibling.

The routing tables are fixed shape_arrays.  Their values enter as scalar
arguments (JSON pointers) that fix the static loop structure.  The table
tensors are also bound as int32 arguments for the fixed-input identity check.
"""
import torch

CHUNK = 128


def reference(X, B, C, delta, cumsum, initial_states,
              chunk_indices, chunk_offsets, seq_chunk_cumsum,
              chunk_index_0, chunk_index_1, chunk_index_2,
              chunk_offset_0, chunk_offset_1, chunk_offset_2,
              seq_chunk_0, seq_chunk_1, seq_chunk_2,
              nheads, ngroups, batch, seqlen, nchunks, sequence_count,
              num_logical_chunks, mode_varlen, has_seq_chunk_cumsum, seq_idx_int64,
              D_mode, has_z, has_initial, write_final_states, checkpoint_state_count):
    indices = [chunk_index_0, chunk_index_1, chunk_index_2]
    offsets = [chunk_offset_0, chunk_offset_1, chunk_offset_2]
    seq_chunks = [seq_chunk_0, seq_chunk_1, seq_chunk_2]
    # Static scope from the scalar kernel arguments.
    if not mode_varlen or not has_seq_chunk_cumsum or seq_idx_int64:
        raise ValueError("binding covers varlen routing through seq_chunk_cumsum")
    if batch != 1 or seqlen != nchunks * CHUNK:
        raise ValueError("varlen packs all tokens in batch row 0; seqlen = nchunks * 128")
    if num_logical_chunks != len(indices) or sequence_count + 1 != len(seq_chunks):
        raise ValueError("binding covers 3 logical chunks and 2 sequences")
    if tuple(chunk_indices.shape) != (num_logical_chunks,) or \
            tuple(seq_chunk_cumsum.shape) != (sequence_count + 1,) or \
            chunk_offsets.shape != chunk_indices.shape:
        raise ValueError("table tensors must match the fixed table sizes")
    if D_mode or has_z or not has_initial or checkpoint_state_count or not write_final_states:
        raise ValueError("binding covers no D, no z, initial states, final states only")
    head_dim, state_dim = X.shape[-1], B.shape[-1]
    x, b, c = X.float(), B.float(), C.float()
    rows = {}
    finals = []
    for sequence in range(sequence_count):
        head_states = []
        for head in range(nheads):
            group = head * ngroups // nheads
            state = initial_states[sequence, head].float()
            for logical in range(seq_chunks[sequence], seq_chunks[sequence + 1]):
                physical = indices[logical]
                offset = offsets[logical]
                limit = CHUNK
                if logical + 1 < num_logical_chunks and indices[logical + 1] == physical:
                    limit = offsets[logical + 1]
                previous = 0.0  # segment base: smem cumsum before the segment is 0
                for row in range(offset, limit):
                    token = physical * CHUNK + row
                    local = row - offset
                    cs = cumsum[logical, head, local:local + 1]
                    dl = delta[logical, head, local:local + 1].float()
                    decay = torch.exp(cs - previous).reshape(1, 1)
                    previous = cs
                    update = (dl * x[0, token, head]).reshape(head_dim, 1) * \
                        b[0, token, group].reshape(1, state_dim)
                    state = decay * state + update
                    rows[(token, head)] = (state * c[0, token, group].reshape(1, state_dim)).sum(dim=-1)
            head_states.append(state)
        finals.append(torch.stack(head_states, dim=0))
    missing = [key for key in ((t, h) for t in range(seqlen) for h in range(nheads)) if key not in rows]
    if missing:
        raise ValueError("binding requires every token row to belong to one segment")
    out = torch.stack([torch.stack([rows[(t, h)] for h in range(nheads)], dim=0)
                       for t in range(seqlen)], dim=0)
    out = out.reshape(1, seqlen, nheads, head_dim).to(torch.bfloat16)
    final = torch.stack(finals, dim=0).to(torch.bfloat16)
    return out, final
