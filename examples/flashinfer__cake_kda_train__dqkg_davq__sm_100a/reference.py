"""cake_kda_train dqkg_davq: the fused chunk-parallel gradient of the packed KDA backward (dq / dk / dg / dbeta
partials, dv, dA_kk and dA_qk) for one (chunk, head) item per persistent CTA iteration.

Program: flashinfer-ai/flashinfer@47e64b0e96c446eaba90e40a1ecdf5891aaef92d
  flashinfer/experimental/cake_kda_train/csrc/cake_kda_train_cfa2684db4d92e37448d_kernel.cu
  (registry record ``dqkg_davq`` of cake_jit.MODULES, stage ``dqkg``; launched by cake_backend._backward_launches
  on the light path of the segmented route, after the adjoint recurrence).

Semantics, from the kernel source (roles compute / load / mma), for every real chunk c < seq_chunk_start[num_seqs]
and head h, token rows T = chunk_bos[c] + t and internal rows r = 64 c + t (t < 64; this binding covers full
chunks, chunk_len == 64):
  DO, V, K, Q = dout[T], v[T], k[T], q[T]; VN, DV2, G = v_new[r], dv2[r], gk[r]; B = beta[r];
  Hc, DHc = h[c, h], dh[c, h] ([K = 128, V = 128] tapes); A = Akk[T] (64 x 64 token rows of the saved set).
  eg = exp2(G); gn = G[63]
  dq  = (DO @ Hc^T) * eg * scale                                   -> dq_f[r]
  dk2 = (VN @ DHc^T) * exp2(gn - G)
  dw  = -bf16(DV2 @ Hc^T);  kgb = bf16(K * eg)
  dgk = exp2(gn) * sum_v bf16(Hc * DHc) + sum_t K * dk2            (per key column)
  dvb = A^T @ DV2;  dkgb = A^T @ dw
  dv  = bf16(dvb * B)                                              -> dv[T]
  db  = sum_j dvb * V + sum_j dkgb * K * eg                        -> db_f[r]
  dg  = Q * dq - K * dk2 + K * eg * dkgb * B, plus dgk on row 63    -> dg_f[r]
  dk  = dkgb * eg * B + dk2                                        -> dk_f[r]
  da  = DV2 @ V^T + dw @ kgb^T;  dab = bf16(da * B[s]) for s < t, else 0
  dAkk = -(A^T @ bf16(dab @ A^T)) for s < t, else 0                 -> dAkk[r]
  dAqk = scale * (DO @ VN^T) for s <= t, else 0                    -> dAqk[r]
"""
import torch


def reference(dout, v_new, dv2, v, h, dh, akk, gk, k, q, beta, scale, num_heads, num_seqs,
              seq_chunk_start_0, seq_chunk_start_1, seq_chunk_start_2,
              chunk_bos_0, chunk_bos_1, chunk_bos_2, chunk_bos_3,
              chunk_len_0, chunk_len_1, chunk_len_2, chunk_len_3):
    starts = [seq_chunk_start_0, seq_chunk_start_1, seq_chunk_start_2]
    chunk_bos = [chunk_bos_0, chunk_bos_1, chunk_bos_2, chunk_bos_3]
    chunk_len = [chunk_len_0, chunk_len_1, chunk_len_2, chunk_len_3]
    nc = starts[num_seqs]
    heads = q.shape[1]
    if heads != num_heads:
        raise ValueError("head count mismatch")
    dq_rows, dk_rows, dg_rows, db_rows, dakk_rows, daqk_rows, dv_rows = [], [], [], [], [], [], []
    idx = torch.arange(64, device=q.device)
    lower = idx.unsqueeze(0) < idx.unsqueeze(1)       # [t, s]: s < t
    lower_eq = idx.unsqueeze(0) <= idx.unsqueeze(1)   # [t, s]: s <= t
    for c in range(nc):
        if chunk_len[c] != 64:
            raise ValueError("binding covers full chunks only")
        t0, r0 = chunk_bos[c], 64 * c
        if t0 != sum(chunk_len[:c]):
            raise ValueError("binding covers chunks laid out in token order")
        dq_h, dk_h, dg_h, db_h, dakk_h, daqk_h, dv_h = [], [], [], [], [], [], []
        for hd in range(heads):
            DO = dout[t0:t0 + 64, hd].float()
            V = v[t0:t0 + 64, hd].float()
            K = k[t0:t0 + 64, hd].float()
            Q = q[t0:t0 + 64, hd].float()
            VN = v_new[r0:r0 + 64, hd].float()
            DV2 = dv2[r0:r0 + 64, hd].float()
            G = gk[r0:r0 + 64, hd].float()
            Bt = beta[r0:r0 + 64, hd].float()
            Hc = h[c, hd].float()
            DHc = dh[c, hd].float()
            A = akk[t0:t0 + 64, hd].float()
            eg = torch.exp2(G)
            gn = G[63]
            dq = (DO @ Hc.transpose(0, 1)) * eg * scale
            dk2 = (VN @ DHc.transpose(0, 1)) * torch.exp2(gn.unsqueeze(0) - G)
            dw = -((DV2 @ Hc.transpose(0, 1)).to(torch.bfloat16).float())
            kgb = (K * eg).to(torch.bfloat16).float()
            # the per-column token sum written out term by term (a 64-term sum over the chunk rows)
            colsum = K[0] * dk2[0]
            for t in range(1, 64):
                colsum = colsum + K[t] * dk2[t]
            dgk = torch.exp2(gn) * (Hc * DHc).to(torch.bfloat16).float().sum(dim=1) + colsum
            At = A.transpose(0, 1)
            dvb = At @ DV2
            dkgb = At @ dw
            dv = (dvb * Bt.unsqueeze(1)).to(torch.bfloat16)
            db = (dvb * V).sum(dim=1) + (dkgb * K * eg).sum(dim=1)
            dg = Q * dq - K * dk2 + K * eg * dkgb * Bt.unsqueeze(1)
            dg = torch.cat([dg[:63], (dg[63] + dgk).unsqueeze(0)], dim=0)
            dk = dkgb * eg * Bt.unsqueeze(1) + dk2
            da = DV2 @ V.transpose(0, 1) + dw @ kgb.transpose(0, 1)
            dab = torch.where(lower, da * Bt.unsqueeze(0), torch.zeros_like(da)).to(torch.bfloat16).float()
            da1b = (dab @ At).to(torch.bfloat16).float()
            dakk = torch.where(lower, -(At @ da1b), torch.zeros_like(da))
            daqk = torch.where(lower_eq, scale * (DO @ VN.transpose(0, 1)), torch.zeros_like(da))
            dq_h.append(dq); dk_h.append(dk); dg_h.append(dg); db_h.append(db)
            dakk_h.append(dakk); daqk_h.append(daqk)
            dv_h.append(dv)
        dq_rows.append(torch.stack(dq_h, 1)); dk_rows.append(torch.stack(dk_h, 1))
        dg_rows.append(torch.stack(dg_h, 1)); db_rows.append(torch.stack(db_h, 1))
        dakk_rows.append(torch.stack(dakk_h, 1)); daqk_rows.append(torch.stack(daqk_h, 1))
        dv_rows.append(torch.stack(dv_h, 1))
    dv_out = torch.cat(dv_rows, 0)
    if dv_out.shape[0] != q.shape[0]:
        raise ValueError("the chunks do not cover every token row of dv")
    return (torch.cat(dq_rows, 0), torch.cat(dk_rows, 0), torch.cat(dg_rows, 0), torch.cat(db_rows, 0),
            torch.cat(dakk_rows, 0), dv_out, torch.cat(daqk_rows, 0))
