"""Block-scaled GEMM references over the original packed storage."""
import torch


def fp4_values(packed):
    low, high = packed & 15, packed >> 4
    nibble = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    exponent, mantissa = (nibble >> 1) & 3, nibble & 1
    magnitude = torch.where(exponent == 0, mantissa.float() * .5,
                            torch.exp2(exponent.float() - 1) * (1 + mantissa.float() * .5))
    return torch.where((nibble & 8) != 0, -magnitude, magnitude)


def scale_offsets(rows, columns, device):
    # CUTLASS detail/sm100_blockscaled_layout.hpp: SfKMajorAtom + Step<2,1>.
    row = torch.arange(rows, device=device)[:, None]
    column = torch.arange(columns, device=device)[None, :]
    return ((row // 128) * ((columns + 3) // 4) * 512 + (column // 4) * 512
            + (row % 32) * 16 + ((row % 128) // 32) * 4 + column % 4)


def block_scales(storage, rows, columns, e8m0):
    values = storage[scale_offsets(rows, columns, storage.device)]
    if e8m0:
        return torch.where(values == 255, float('nan'), torch.exp2(values.float() - 127))
    exponent, mantissa = (values >> 3) & 15, values & 7
    normal = torch.exp2(exponent.float() - 7) * (1 + mantissa.float() / 8)
    decoded = torch.where(exponent == 0, mantissa.float() / 512, normal)
    return torch.where((values & 127) == 127, float('nan'), decoded)


def product(A, B, scale_a, scale_b, *, packed, block, e8m0):
    a, b = (fp4_values(A), fp4_values(B)) if packed else (A.float(), B.float())
    groups = a.shape[1] // block
    sa = block_scales(scale_a, a.shape[0], groups, e8m0)
    sb = block_scales(scale_b, b.shape[0], groups, e8m0)
    a = (a.reshape(a.shape[0], groups, block) * sa.unsqueeze(-1)).reshape(a.shape)
    b = (b.reshape(b.shape[0], groups, block) * sb.unsqueeze(-1)).reshape(b.shape)
    return a @ b.T
