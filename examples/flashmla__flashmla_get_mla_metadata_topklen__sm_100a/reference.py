"""FlashMLA get_mla_metadata_kernel<true, false, false> (split-KV tile scheduler metadata).

Source (FlashMLA 395cb3afbb331fef7b1b21e9340861b9436b42d8):
  csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu:43-57 (per request:
    cur_s_k = topk_length[i], 0 -> 1; num_blocks = (cur_s_k-1)/block_size_topk + 1; last block = num_blocks-1;
    total = sum(num_blocks + fixed_overhead)), :64-116 (payload = ceil(total/num_sm_parts) + fixed_overhead and the
    greedy partition loop), :119-121 (num_splits copied out); csrc/params.h:84-95 (DecodingSchedMeta: begin_req,
    end_req, begin_block, end_block, begin_split, is_first_req_splitted, is_last_req_splitted, _pad).
The while loop is written without data-dependent Python control: every partition runs b+1 masked steps
(each step either consumes a whole request or ends the partition), and all choices are torch.where selects.
Only DecodingSchedMeta words 0-6 are returned (_pad is never written by the kernel).
"""
import torch


def reference(topk_length, block_size_topk, fixed_overhead_num_blocks, num_sm_parts):
    b = topk_length.shape[0]
    i32 = torch.int32
    dev = topk_length.device
    cur = torch.where(topk_length == 0, torch.ones_like(topk_length), topk_length)
    last_block = torch.floor_divide(cur - 1, torch.ones_like(cur) * block_size_topk)   # (uint32)(cur-1)/bs, cur >= 1
    num_blocks = last_block + 1
    total = (num_blocks + fixed_overhead_num_blocks).sum()
    payload = torch.floor_divide(total + (num_sm_parts - 1), torch.ones_like(total) * num_sm_parts) + fixed_overhead_num_blocks
    zero = torch.zeros((), dtype=i32, device=dev)
    req, blk, nsplit, cum = zero, zero, zero, zero
    pos = torch.arange(b + 1, dtype=i32, device=dev)
    num_splits = torch.zeros(b + 1, dtype=i32, device=dev)
    rows = []
    for _ in range(int(num_sm_parts)):
        begin_req, begin_blk, begin_split = req, blk, nsplit
        is_first = (blk != 0).to(i32)
        remain = payload.to(i32)
        active = torch.ones((), dtype=torch.bool, device=dev)
        for _ in range(b + 1):
            live = active & (req < b)
            nb = num_blocks.index_select(0, req.clamp(max=b - 1).reshape(1).long()).reshape(())
            rem_blocks = nb - blk
            consume = live & (remain >= rem_blocks + fixed_overhead_num_blocks)
            partial = live & (remain < rem_blocks + fixed_overhead_num_blocks) & (remain - fixed_overhead_num_blocks > 0)
            cum = torch.where(consume, cum + nsplit + 1, cum)
            num_splits = torch.where(consume & (pos == req + 1), cum, num_splits)
            new_blk = torch.where(consume, zero, torch.where(partial, blk + remain - fixed_overhead_num_blocks, blk))
            nsplit = torch.where(consume, zero, torch.where(partial, nsplit + 1, nsplit))
            remain = torch.where(consume, remain - rem_blocks - fixed_overhead_num_blocks,
                                 torch.where(partial, zero, remain))
            req = torch.where(consume, req + 1, req)
            blk = new_blk
            active = consume
        end_req = torch.where(blk > 0, req, req - 1)
        prev_last = last_block.index_select(0, (req - 1).clamp(min=0, max=b - 1).reshape(1).long()).reshape(())
        end_blk = torch.where(blk > 0, blk, prev_last + 1)
        end_last = last_block.index_select(0, end_req.clamp(min=0, max=b - 1).reshape(1).long()).reshape(())
        is_last = (end_blk != end_last + 1).to(i32)
        same = begin_req == end_req
        both = ((is_first != 0) | (is_last != 0)).to(i32)
        is_first = torch.where(same, both, is_first)
        is_last = torch.where(same, both, is_last)
        rows.append(torch.stack([begin_req, end_req, begin_blk, end_blk, begin_split, is_first, is_last]).to(i32))
    return torch.stack(rows), num_splits
