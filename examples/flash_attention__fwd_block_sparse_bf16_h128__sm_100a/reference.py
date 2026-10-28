# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_sm100.py#L426-L450
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_sm100.py#L2159-L2300
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/block_sparse_utils.py#L60-L77
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/block_sparse_utils.py#L150-L160
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/block_sparse_utils.py#L620-L675
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/block_sparse_utils.py#L833-L948
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/block_sparse_utils.py#L951-L1155
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/block_sparsity.py#L198-L211
# Source: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/mask.py#L615-L672
import torch


def reference(Q, K, V, mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx,
              scale_log2, sparse_block_size_kv):
    # Q, K, V, O use the logical (batch, head, sequence, dim) layout.
    # Block-sparse metadata is [batch, head, m_block(, slot)]; the single sparse
    # Q block (m_block 0) spans all 256 query rows of this scenario.
    columns = K.shape[2]
    key = torch.arange(columns, dtype=torch.int32, device=Q.device).reshape(1, 1, columns)
    score = (Q.float() @ K.float().transpose(-1, -2)) * scale_log2
    # Common row scaling (running max) cancels under ideal-real semantics.
    exp_score = torch.exp2(score)
    # Slot j of a list is used iff j < count; coarse block c covers keys
    # [c * sparse_block_size_kv, (c + 1) * sparse_block_size_kv).
    # mask_mod is None in this build, so masked-list blocks get only the seqlen_k
    # mask, exactly like full-list blocks; a block listed twice contributes twice.
    weight = None
    for count, index in ((mask_block_cnt, mask_block_idx), (full_block_cnt, full_block_idx)):
        for slot in range(index.shape[-1]):
            start = index[:, :, 0, slot:slot + 1] * sparse_block_size_kv
            active = ((count[:, :, 0:1] > slot) & (key >= start)
                      & (key < start + sparse_block_size_kv))
            term = torch.where(active.unsqueeze(2), exp_score, 0.0)
            weight = term if weight is None else weight + term
    total = weight.sum(dim=-1, keepdim=True)
    # Tiles with no visited key write O = 0 (block_sparse_utils.py empty-tile path).
    output = (weight @ V.float()) / torch.where(total > 0, total, 1.0)
    return output.to(torch.bfloat16)
