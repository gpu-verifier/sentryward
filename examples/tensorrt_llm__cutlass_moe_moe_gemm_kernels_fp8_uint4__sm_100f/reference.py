"""Mathematical reference; source: https://github.com/NVIDIA/TensorRT-LLM/blob/ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_uint4.cu ; https://github.com/NVIDIA/TensorRT-LLM/blob/ef567dd9b8919f6ee8b2da1f6f6d0c90206a78f8/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"""
import torch


def _rows(row_end_0, row_end_1, row_end_2, row_end_3):
    # Expert e owns rows [last_row[e-1], last_row[e]) (last_row_for_problem, fixed by the input's shape array).
    ends = (row_end_0, row_end_1, row_end_2, row_end_3)
    return list(zip((0,) + ends[:-1], ends))


def _weights(expert_weights, experts, k, n):
    # ColumnMajorTileInterleave<128, 2> with the W4_AFP8 row permutation and nibble interleave: the nibble of
    # W[k][n] is t = (n//2)*2K + (k//128)*256 + (n%2)*128 + 32*((k//32)%4) + 8*((k%16)//4) + 4*(k%2)
    # + 2*((k%32)//16) + (k%4)//2 (low nibble of byte t//2 if t is even), stored as w + 8.
    packed = expert_weights.to(torch.int32).reshape(experts, n // 2, k // 128, 2, 4, 4, 2, 2)
    nibbles = torch.stack(((packed & 15) - 8, (packed >> 4) - 8), dim=-1)
    return nibbles.permute(0, 2, 4, 7, 5, 8, 6, 1, 3).reshape(experts, k, n).to(torch.float16)


def reference(activations, expert_weights, weight_scales, row_end_0, row_end_1, row_end_2, row_end_3):
    experts, n = weight_scales.shape
    weights = _weights(expert_weights, experts, activations.shape[1], n)
    parts = []
    for e, (start, end) in enumerate(_rows(row_end_0, row_end_1, row_end_2, row_end_3)):
        # B is dequantized in fp16 and converted to e4m3 (cvt.rn.satfinite) before the e4m3 MMA.
        b = (weights[e] * weight_scales[e][None, :]).float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        acc = activations[start:end].float() @ b.float()
        parts.append(torch.clamp(acc, min=0.0))
    return torch.cat(parts).to(torch.float16)
