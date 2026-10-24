"""Structured-sparse GEMM D = A @ B^T with A given as (A.compressed, E.metadata).

Sources (CUTLASS 0efeafbafc16d5b8b17fd74bbac79674325b38a2):
  include/cutlass/gemm/kernel/sm100_sparse_gemm_tma_warpspecialized.hpp
  include/cutlass/gemm/collective/builders/sm1xx_sparse_config.inl:106-129,236-311 (chunks, TensorEAtom_*, fill_layoutA/E)
  include/cutlass/transform/kernel/sm90_sparse_gemm_compressor.hpp:255-283,430-483 (2:4 metadata encoding)
  python/cutlass_library/generator.py (GenerateSM100_SparseTensorOp_16b_UMMA_gemm)
Kernel: cutlass3x_sm100_sptensorop_spgemm_f16_f16_f32_void_f16_128x128x64_4x4x1_0_tnt_align16_1sm,
alpha = 1 (input.json), C void.
"""
import torch

# kind: (logical A elements per chunk, logical K per metadata atom, logical A elements per E byte)
_E_ATOM = {'f16': (4, 128, 8), 'f8f6f4': (4, 128, 8), 'f4': (8, 256, 16)}


def _metadata_offsets(rows, chunks, kind, device):
    chunk, atom_k, per_byte = _E_ATOM[kind]
    m = torch.arange(rows, device=device).reshape(rows, 1)
    k = torch.arange(chunks, device=device).reshape(1, chunks) * chunk  # first logical K of each chunk
    mi, ki = m % 128, k % atom_k
    if kind == 'f16':
        # TensorEAtom_MMA_F16 ((8,2,8),(16,2,4)):((128,16,2048),(1,1024,32)) in logical units, 8 per byte.
        byte = ki // 8
        inner = ((mi % 8) * 16 + ((mi // 8) % 2) * 2 + (mi // 16) * 256
                 + byte % 2 + ((byte // 2) % 2) * 128 + (byte // 4) * 4)
    else:
        # TensorEAtom_MMA_S8orF8F6F4 (128,128):(128,1) and TensorEAtom_MMA_F4 (128,256):(256,1).
        inner = mi * 16 + ki // per_byte
    rows_aligned = (rows + 127) // 128 * 128
    # fill_layoutE: atoms first along M (stride cosize), then along K (stride M_aligned * AtomK).
    offset = inner + (m // 128) * (128 * atom_k // per_byte) + (k // atom_k) * (rows_aligned * atom_k // per_byte)
    return offset, (k // chunk) % 2 == 1  # odd chunks use the high nibble of the byte


def _metadata_indices(E, rows, chunks, kind):
    offset, high = _metadata_offsets(rows, chunks, kind, E.device)
    byte = E[offset]
    nibble = torch.where(high, byte >> 4, byte & 15)
    # MetadataOneChunk2to4: bits[1:0] = slot of stored entry 0, bits[3:2] = slot of stored entry 1.
    return nibble & 3, (nibble >> 2) & 3


def sparse_a(values, E, kind):
    """Dense logical A [rows, K] from decoded compressed values [rows, K/2] and metadata bytes E."""
    rows = values.shape[0]
    width = 2 if kind == 'f4' else 1  # F4 metadata selects element pairs (4:8 pairwise)
    chunks = values.shape[1] // (2 * width)
    first, second = _metadata_indices(E, rows, chunks, kind)
    stored = values.reshape(rows, chunks, 2, width)
    slot = torch.arange(4, device=values.device).reshape(1, 1, 4, 1)
    # Each stored entry multiplies the B element at its metadata slot (additive placement).
    dense = (torch.where(first.unsqueeze(-1).unsqueeze(-1) == slot, stored[:, :, 0:1], 0.0)
             + torch.where(second.unsqueeze(-1).unsqueeze(-1) == slot, stored[:, :, 1:2], 0.0))
    return dense.reshape(rows, chunks * 4 * width)


def reference(A, E, B):
    a = sparse_a(A.float(), E, 'f16')
    return (a @ B.float().T).to(torch.float16)
