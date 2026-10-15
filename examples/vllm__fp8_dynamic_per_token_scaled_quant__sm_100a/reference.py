from examples._reference.pytorch_models import rowwise_gemm, rope_positions, rope_cache, fp8_per_token, softmax


def reference(input):
    return fp8_per_token(input)
