"""NVFP4 linear quantization, including the INTERNAL rounded E4M3 scale.

Exporter integration uses the accompanying staged lowering.py extensions.
Do not erase the internal float8 conversion as an output cast.
"""
import torch


def reference(A, B):
    # A is the scalar global scale; B consists of groups of 16 half elements.
    x = B.float()
    absolute = x.abs()
    # PTX max (without .NaN) ignores a NaN when the other operand is numeric.
    non_nan_max = torch.where(absolute == absolute, absolute, -float("inf")).amax(dim=-1)
    maximum = torch.where(non_nan_max == -float("inf"), float("nan"), non_nan_max)
    raw_scale = A * (maximum / 6.0)
    scale = raw_scale.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    decoded_scale = scale.float()
    multiplier = torch.where(decoded_scale == 0, 0.0,
                             1.0 / (decoded_scale * (1.0 / A)))
    scaled = x * multiplier[:, None]
    magnitude = scaled.abs()
    # E2M1 positive magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6.
    # Midpoint ties select an even encoded mantissa (round-to-nearest-even).
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    code = torch.where(magnitude > .25, 1, code)
    code = torch.where(magnitude >= .75, 2, code)
    code = torch.where(magnitude > 1.25, 3, code)
    code = torch.where(magnitude >= 1.75, 4, code)
    code = torch.where(magnitude > 2.5, 5, code)
    code = torch.where(magnitude >= 3.5, 6, code)
    code = torch.where(magnitude > 5.0, 7, code)
    code = code.to(torch.uint8) | (torch.signbit(scaled).to(torch.uint8) * 8)
    # PTX cvt.satfinite.e2m1x2: NaN converts to positive MAX_NORM.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    code = torch.where(scaled != scaled, 7, code).to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] * 16)
    return packed, scale.view(torch.uint8)
