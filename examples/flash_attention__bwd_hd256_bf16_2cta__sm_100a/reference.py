# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py#L156-L276
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py#L1975-L2022
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py#L2064-L2194
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py#L2196-L2290
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/sm100_hd256_2cta_fmha_backward.py#L115-L225
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/mask.py#L1661-L1711
import torch

# The kernel forms scale * log2(e) with the float32 literal 0f3FB8AA3B.
LOG2E_F32 = 1.4426950216293335


def reference(Q, K, V, dO, LSE, sum_odo, scale):
    # Q, K, V, dO, dQ use the logical (batch, head, sequence, dim) layout.
    # LSE is the preprocess output lse_log2 (log2 domain); sum_odo is dPsum.
    # scale is the kernel argument scale_softmax (param 18), bound from input.json.
    q, k, v, do = Q.float(), K.float(), V.float(), dO.float()
    probability = torch.exp2((q @ k.transpose(-1, -2)) * (scale * LOG2E_F32) - LSE.unsqueeze(-1))
    derivative = do @ v.transpose(-1, -2)
    # dS is scaled by softmax_scale and rounded to bf16 before the dS @ K MMA.
    ds = ((derivative - sum_odo.unsqueeze(-1)) * probability * scale).to(Q.dtype).float()
    # dQ is final (no postprocess): fp32 TMEM accumulation, bf16 TMA store (overwrite).
    return (ds @ k).to(Q.dtype)
