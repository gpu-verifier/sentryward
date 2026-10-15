from examples._reference.pytorch_models import rowwise_gemm, rope_positions, rope_cache, fp8_per_token, softmax


def reference(A, B, x_scale, w_scale):
    return rowwise_gemm(A, B, x_scale, w_scale)
