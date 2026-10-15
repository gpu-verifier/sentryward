from examples._reference.pytorch_models import fused_qk_rope, attention_backward


def reference(qkv, q_weight, k_weight, position_ids, cos_sin_cache, epsilon):
    return fused_qk_rope(qkv, q_weight, k_weight, position_ids, cos_sin_cache, epsilon)
