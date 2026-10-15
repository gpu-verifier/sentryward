from examples._reference.pytorch_models import rowwise_gemm, rope_positions, rope_cache, fp8_per_token, softmax


def reference(A, B, scale_a, scale_b):
    return rowwise_gemm(A, B, scale_a, scale_b)
