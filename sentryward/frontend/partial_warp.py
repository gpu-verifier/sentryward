"""Partial-warp launches for kernels that never synchronize or communicate within a warp.

The native verifier models a CTA as complete 32-lane warps and refuses a block whose thread
count is not a multiple of 32. A kernel without barriers, warp collectives or lane/warp
identities cannot observe the difference between a thread that was never launched and one that
exits before doing anything. Such a kernel is verified at a padded block: the z dimension grows
until the block fills whole warps (x and y, and so the thread-index mapping, are unchanged), the
padded threads (tid.z >= Z) exit at entry, and reads of %ntid.z read the launched Z.

Anything that could see the padding keeps the native refusal: barriers (bar, barrier, mbarrier),
warp collectives (shfl, vote, match, redux, activemask, elect), lane and warp identities
(%laneid, %warpid, %lanemask_*), asynchronous and tensor-core operations, calls, and %ntid used
other than per dimension. The guard is inserted on the line of the entry's opening brace, so
the line numbers of every witness are those of the original PTX.
"""
from __future__ import annotations

import re

_REFUSED = re.compile(
    r"\b(bar|barrier|mbarrier|shfl|vote|match|redux|activemask|elect|wgmma|tcgen05|cp\.async|"
    r"griddepcontrol|fence\.proxy|call)\b|%laneid|%warpid|%nwarpid|%lanemask_\w+|%ntid(?!\.[xyz])")


def _entry_body(ptx: str, entry: str | None):
    """(start, end) of the entry's body: the offsets just after its '{' and of its matching '}'."""
    names = re.findall(r"\.entry\s+([\w$]+)", ptx)
    if entry is None:
        if len(names) != 1:
            return None
        entry = names[0]
    m = re.search(r"\.entry\s+" + re.escape(entry) + r"\b", ptx)
    if m is None:
        return None
    open_ = ptx.find("{", ptx.find(")", m.end()))
    if open_ < 0:
        return None
    depth = 0
    for i in range(open_, len(ptx)):
        if ptx[i] == "{":
            depth += 1
        elif ptx[i] == "}":
            depth -= 1
            if depth == 0:
                return open_ + 1, i
    return None


def pad(ptx: str, kernel_input: dict):
    """The rewritten PTX and input for a partial-warp launch it can verify, or None.

    None also when the launch fills whole warps: then nothing needs rewriting.
    """
    launch = kernel_input.get("launch") or {}
    block = list(launch.get("block_dim") or [])
    if len(block) != 3 or not all(isinstance(n, int) and n > 0 for n in block):
        return None
    x, y, z = block
    if (x * y * z) % 32 == 0:
        return None
    body = _entry_body(ptx, kernel_input.get("entry"))
    if body is None:
        return None
    start, end = body
    text = ptx[start:end]
    if _REFUSED.search(text):
        return None
    padded_z = z
    while (x * y * padded_z) % 32:
        padded_z += 1
    if x * y * padded_z > 1024:
        return None
    guard = (" .reg .pred %sw_partial_p; .reg .b32 %sw_partial_t; mov.u32 %sw_partial_t, %tid.z;"
             f" setp.ge.u32 %sw_partial_p, %sw_partial_t, {z}; @%sw_partial_p ret;")
    rewritten = ptx[:start] + guard + re.sub(r"%ntid\.z\b", str(z), text) + ptx[end:]
    padded = dict(kernel_input)
    padded["launch"] = dict(launch, block_dim=[x, y, padded_z])
    warning = {
        "code": "partial_warp_padded",
        "message": (f"block {block} is not a whole number of warps: verified at {[x, y, padded_z]}, "
                    f"threads with tid.z >= {z} exiting at entry (the kernel has no barriers or warp collectives)"),
        "details": {"launched_block_dim": block, "verified_block_dim": [x, y, padded_z]},
    }
    return rewritten, padded, warning
