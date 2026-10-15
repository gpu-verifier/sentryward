from examples._reference.pytorch_models import rowwise_gemm, rope_positions, rope_cache, fp8_per_token, softmax


def reference(q, k, pos_ids, reciprocal_theta, reciprocal_scale):
    return rope_positions(q, k, pos_ids, reciprocal_theta, reciprocal_scale)
