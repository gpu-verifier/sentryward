"""Warp-specialized Triton TMA matmul: C = A @ B^T in fp16 (M = N = K = 512).

Source: https://github.com/triton-lang/triton/blob/1826d8ef0baa70686860658f5ab5dd7cb0d3b163/python/test/unit/language/test_warp_specialization.py
(L189-L253 `_compute_pid`, `_maybe_tma_load`, `matmul_tma_ws_kernel`; the test checks C against
cublas.matmul(A, B) of A[M, K] and B[N, K]).
"""
import torch


def reference(A, B):
    # tl.dot(a, b.T, accumulator) over all K tiles with an f32 accumulator, then .to(float16).
    # GROUP_SIZE_M only permutes the tile-to-CTA assignment; 16 CTAs cover all of C.
    return (A.float() @ B.float().T).to(torch.float16)
