from examples._reference.pytorch_models import fused_qk_rope, attention_backward


def reference(Q, K, V, dO, LSE, dPsum, dQaccum, scale, scale_log2):
    return attention_backward(Q, K, V, dO, LSE, dPsum, dQaccum, scale, scale_log2)
