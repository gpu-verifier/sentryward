"""Mathematical reference; sources (pinned):
https://github.com/vllm-project/vllm/blob/605c3ddcbaaee06a2687a961ae7be51aec5dddc0/csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu
https://github.com/vllm-project/vllm/blob/605c3ddcbaaee06a2687a961ae7be51aec5dddc0/csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x.cuh
https://github.com/vllm-project/vllm/blob/605c3ddcbaaee06a2687a961ae7be51aec5dddc0/csrc/libtorch_stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp

cutlass_3x_group_gemm<e4m3, bf16, Sm100, ScaledEpilogueArray, 128x16x128, 1x1x1, ..., swap_ab = true>
(sm100_fp8_config_M64). The kernel's A operand is the expert weight (b_ptrs), its B operand the
activations (a_ptrs); D is LayoutD_Transpose (column-major). ScaledEpilogueArray computes
  D_g(m, n) = ScaleA_g(m) * (ScaleB_g(n) * sum_k A_g(m, k) * B_g(n, k))
where ScaleA is the per-group ColOrScalar array at param +1280 (b_scales_ptrs with swap_ab) and ScaleB
the RowOrScalar array at +1296 (a_scales_ptrs). The input sets both broadcast flags (+1288, +1304) to
false, so each group uses the scalar element 0 of its two scale buffers.
"""
import torch


def reference(A0, A1, A2, A3, B0, B1, B2, B3, scale_a0, scale_a1, scale_a2, scale_a3, scale_b0, scale_b1, scale_b2, scale_b3, table_problem_shapes, table_scheduler_problem_shapes, table_mainloop_dA_array, table_mainloop_dB_array, table_c_strides_param1952):
    # table_problem_shapes, table_scheduler_problem_shapes, table_mainloop_dA_array, table_mainloop_dB_array, table_c_strides_param1952: fixed shape-array tables, bound only so the native fixed-input
    # context finds a common identity on both binding sides (fixed_input_context.hpp decode_verified);
    # their values enter through the binding shapes and scalars, not through this computation.
    outputs = []
    for a, b, sa, sb in zip((A0, A1, A2, A3), (B0, B1, B2, B3), (scale_a0, scale_a1, scale_a2, scale_a3),
                            (scale_b0, scale_b1, scale_b2, scale_b3)):
        acc = a.float() @ b.float().T
        outputs.append((sa[0] * (sb[0] * acc)).to(torch.bfloat16))
    return tuple(outputs)
