"""Mathematical reference; source (pinned):
https://github.com/vllm-project/vllm/blob/605c3ddcbaaee06a2687a961ae7be51aec5dddc0/csrc/libtorch_stable/moe/topk_softmax_kernels.cu

topkGating<VPT 4, 64 experts, 4 warps, 16-byte loads, warp 32, int indices, float input, SCORING_SOFTMAX> (L278-594),
with finished, bias and is_padding null. Per row r:
  p = softmax(input[r])                                   (max-subtracted exp, times 1/sum)
  for j < k: e_j = argmax over the remaining choices, ties to the lowest expert index (strict > within a thread,
             "> or (== and lower index)" in the butterfly reduce); the chosen expert's choice value becomes -10000
  output[r, j]      = p[e_j] * routed_scaling_factor / (renormalize ? (S > 0 ? S : 1) : 1),  S = sum_j p[e_j]
  indices[r, j]     = start <= e_j < end ? e_j - start : NUM_EXPERTS
  source_rows[r, j] = j * num_rows + r
The kernel's NaN/Inf-to-zero clamp of p (L466-472) is the identity on ideal real values and is omitted.
"""
import torch


def reference(input, num_rows, k, start_expert, end_expert, renormalize, routed_scaling_factor):
    x = input.float()
    rows, experts = x.shape
    e = torch.exp(x - x.amax(dim=-1, keepdim=True))
    p = e / e.sum(dim=-1, keepdim=True)
    column = torch.arange(experts, device=x.device)
    choice = p
    weights, chosen = [], []
    for _ in range(k):
        best = choice.amax(dim=-1, keepdim=True)
        expert = torch.where(choice == best, column, experts).amin(dim=-1)           # lowest index among maxima
        weights.append(p.gather(1, expert[:, None])[:, 0])
        chosen.append(expert)
        choice = torch.where(column[None, :] == expert[:, None], -10000.0, choice)
    weight = torch.stack(weights, dim=-1)
    expert = torch.stack(chosen, dim=-1)
    if renormalize:
        total = weight.sum(dim=-1, keepdim=True)
        weight = weight * routed_scaling_factor / torch.where(total > 0, total, 1.0)
    else:
        weight = weight * routed_scaling_factor
    used = (expert >= start_expert) & (expert < end_expert)
    indices = torch.where(used, expert - start_expert, experts).to(torch.int32)
    source_rows = (torch.arange(k, device=x.device)[None, :] * num_rows
                   + torch.arange(rows, device=x.device)[:, None]).to(torch.int32)
    return weight, indices, source_rows
