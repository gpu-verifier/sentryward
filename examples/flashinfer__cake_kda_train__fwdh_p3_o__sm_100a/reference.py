"""cake_kda_train fwdh_p3_o: pass 2 of the segmented KDA forward state recurrence with the fused output.

Program: flashinfer-ai/flashinfer@47e64b0e96c446eaba90e40a1ecdf5891aaef92d
  flashinfer/experimental/cake_kda_train/csrc/cake_kda_train_9931cdff5dfd86c687c7_kernel.cu
  (registry record ``fwdh_p3_o`` of cake_jit.MODULES; launched by cake_backend._forward_recurrence with use_h0 = 1,
  write_tapes = 0, write_ht = 0 on the saving forward of the segmented route).

Semantics, from the kernel source (roles compute / decayw / load / mma):
  block table ``seg_table`` (cake_backend._tables): entries [0, nchain] are the chains' first segments, then per
  chain its segment starts and end chunk.  With ``seg_arg < 0`` there are ``nchain = -seg_arg`` chains and a
  segment's chain is the number of j in [1, nchain) with seg >= seg_table[j]; segment ``i`` walks chunks
  [seg_table[nchain + 1 + i + chain], seg_table[nchain + 2 + i + chain]).
  For every segment i and head h, starting from S = h0[i, h] (fp32 [K = 128, V = 128]), per chunk c of the walk,
  internal rows r = 64 c .. 64 c + 63:
    hs  = bf16(S)
    v'  = bf16(u[r] - w[r] @ hs)                        (the chunk's v_new)
    o   = scale * (qg[r] @ hs) + aqk[r] @ v'           (written as bf16 to out[chunk_bos[c] + t], t < chunk_len[c])
    S   = exp2(gk[64 c + 63])[:, None] * S + kg[r]^T @ v'
  The tapes (h, v_new) and the final states are not written in this launch (write_tapes = write_ht = 0).
Reference-level contract: w, kg, u, qg are the forward WY / gated rows of the intra and wy_fwd programs, gk the
chunk-local cumulative log2 gate, aqk the masked intra-chunk attention (64 columns), h0 the carried segment states
of seg_carry; all are inputs of this launch.
"""
import torch


def _walk(w, kg, u, gk, qg, aqk, h0, scale, num_segments, seg_arg, table, chunk_bos, chunk_len, total_tokens):
    heads = w.shape[1]
    if seg_arg >= 0:
        raise ValueError("binding covers the proportional block table (seg_arg = -num_docs)")
    nchain = -seg_arg
    if table[nchain] != num_segments or len(table) != num_segments + 2 * nchain + 1:
        raise ValueError("block table does not describe num_segments segments")
    rows = [None] * total_tokens
    for seg in range(num_segments):
        chain = sum(1 for j in range(1, nchain) if seg >= table[j])
        blk = nchain + 1 + seg + chain
        c0, c1 = table[blk], table[blk + 1]
        for h in range(heads):
            state = h0[seg, h].float()
            for c in range(c0, c1):
                r0 = 64 * c
                hs = state.to(torch.bfloat16).float()
                vp = w[r0:r0 + 64, h].float() @ hs
                vnew = (u[r0:r0 + 64, h].float() - vp).to(torch.bfloat16).float()
                o = scale * (qg[r0:r0 + 64, h].float() @ hs) + aqk[r0:r0 + 64, h].float() @ vnew
                for t in range(chunk_len[c]):
                    tok = chunk_bos[c] + t
                    if rows[tok] is None:
                        rows[tok] = [None] * heads
                    rows[tok][h] = o[t]
                decay = torch.exp2(gk[r0 + 63, h].float())
                state = decay[:, None] * state + kg[r0:r0 + 64, h].float().transpose(0, 1) @ vnew
    if any(r is None or any(x is None for x in r) for r in rows):
        raise ValueError("the walks do not cover every token row of out")
    return torch.stack([torch.stack(r, 0) for r in rows], 0).to(torch.bfloat16)


def reference(w, kg, u, gk, qg, aqk, h0, scale, num_segments, seg_arg_bits, total_tokens,
              seg_table_0, seg_table_1, seg_table_2, seg_table_3, seg_table_4, seg_table_5, seg_table_6, seg_table_7, seg_table_8, seg_table_9, seg_table_10, seg_table_11, seg_table_12, seg_table_13, seg_table_14, seg_table_15, seg_table_16, seg_table_17, seg_table_18, seg_table_19, seg_table_20, seg_table_21, seg_table_22, seg_table_23, seg_table_24, seg_table_25, seg_table_26, seg_table_27, seg_table_28, seg_table_29, seg_table_30, seg_table_31, seg_table_32, seg_table_33, seg_table_34, seg_table_35, seg_table_36, seg_table_37, seg_table_38, seg_table_39, seg_table_40, seg_table_41, seg_table_42, seg_table_43, seg_table_44, seg_table_45, seg_table_46, seg_table_47, seg_table_48, seg_table_49, seg_table_50, seg_table_51, seg_table_52, seg_table_53, seg_table_54, seg_table_55, seg_table_56, seg_table_57, seg_table_58, seg_table_59, seg_table_60, seg_table_61, seg_table_62, seg_table_63, seg_table_64, seg_table_65, seg_table_66, seg_table_67, seg_table_68, seg_table_69, seg_table_70, seg_table_71, seg_table_72, seg_table_73, seg_table_74, seg_table_75, seg_table_76, seg_table_77, seg_table_78,
              chunk_bos_0, chunk_bos_1, chunk_bos_2, chunk_bos_3,
              chunk_len_0, chunk_len_1, chunk_len_2, chunk_len_3):
    # the int seg_arg reaches the kernel as a .u32 parameter: reinterpret its 32-bit pattern
    seg_arg = seg_arg_bits - (1 << 32) if seg_arg_bits >= (1 << 31) else seg_arg_bits
    table = [seg_table_0, seg_table_1, seg_table_2, seg_table_3, seg_table_4, seg_table_5, seg_table_6, seg_table_7, seg_table_8, seg_table_9, seg_table_10, seg_table_11, seg_table_12, seg_table_13, seg_table_14, seg_table_15, seg_table_16, seg_table_17, seg_table_18, seg_table_19, seg_table_20, seg_table_21, seg_table_22, seg_table_23, seg_table_24, seg_table_25, seg_table_26, seg_table_27, seg_table_28, seg_table_29, seg_table_30, seg_table_31, seg_table_32, seg_table_33, seg_table_34, seg_table_35, seg_table_36, seg_table_37, seg_table_38, seg_table_39, seg_table_40, seg_table_41, seg_table_42, seg_table_43, seg_table_44, seg_table_45, seg_table_46, seg_table_47, seg_table_48, seg_table_49, seg_table_50, seg_table_51, seg_table_52, seg_table_53, seg_table_54, seg_table_55, seg_table_56, seg_table_57, seg_table_58, seg_table_59, seg_table_60, seg_table_61, seg_table_62, seg_table_63, seg_table_64, seg_table_65, seg_table_66, seg_table_67, seg_table_68, seg_table_69, seg_table_70, seg_table_71, seg_table_72, seg_table_73, seg_table_74, seg_table_75, seg_table_76, seg_table_77, seg_table_78]
    chunk_bos = [chunk_bos_0, chunk_bos_1, chunk_bos_2, chunk_bos_3]
    chunk_len = [chunk_len_0, chunk_len_1, chunk_len_2, chunk_len_3]
    return (_walk(w, kg, u, gk, qg, aqk, h0, scale, num_segments, seg_arg, table, chunk_bos, chunk_len,
                  total_tokens),)
