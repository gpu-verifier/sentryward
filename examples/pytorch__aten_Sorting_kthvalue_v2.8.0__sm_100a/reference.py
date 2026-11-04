# Source: https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/cuda/Sorting.cu
# Source: https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/cuda/Sorting.cpp
# Source: https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/_torch_docs.py
# Source: https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/test_sort_and_select.py
#
# torch.kthvalue(input, k, dim=-1, keepdim=False) at v2.8.0: the k-th smallest value of each slice along the last
# dimension (k is 1-based, _torch_docs.py L5660-L5672) and an index holding it; the value keeps the input dtype and
# the index is int64. test_kthvalue (L1077-L1131) checks the value against torch.sort(x).values[..., k - 1] and, on
# tie-free random input, the index against torch.sort(x).indices[..., k - 1]; with repeated values it checks only the
# value. The v2.8.0 documentation says that with several valid k-th values a CUDA tensor "may nondeterministically
# return indices for any of them" (L5674-L5677). A function reference must pick one of them: this one returns the
# smallest such index (the choice upstream later made deterministic, #165762). It equals the v2.8.0 oracle on every
# tie-free slice; on a slice with ties the v2.8.0 kernel may store any of the valid indices.
import torch


def reference(input, k):
    x = input
    n = x.shape[-1]
    candidate = x.unsqueeze(-1)
    other = x.unsqueeze(-2)
    below = (other < candidate).sum(dim=-1)
    at_most = (other <= candidate).sum(dim=-1)
    is_kth = (below < k) & (at_most >= k)
    value = torch.where(is_kth, x, float("-inf")).amax(dim=-1)
    position = torch.arange(n, device=x.device)
    index = torch.where(x == value.unsqueeze(-1), position, n).amin(dim=-1)
    return value, index
