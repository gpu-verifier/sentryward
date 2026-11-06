"""Same-value write pairs in a storage whose conflict order is reduced.

Lane 0 of several warps stores to one shared word. Each pair of those stores that is unordered covers the same
bytes, so the race search accepts it as a same-value obligation. The search must find every such pair, also when an
obligation breaks the chain of order constraints that certifies its neighbours: the obligation groups and site pairs
it reports are those of all unordered pairs. Without a reference memory elimination still runs and decides them:
stores of different values are unproven (potential_race), equal ones proven (race_free, equivalence skipped).

These runs need the native executable; they are skipped when it is not built.
"""
import json
import tempfile
import unittest
from pathlib import Path

from sentryward.api import verify
from sentryward.frontend.ptx_native import native_executable

PROLOGUE = """.version 8.0
.target sm_100a
.address_size 64

.visible .entry {entry}()
.reqntid {threads}
{{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<8>;
	.shared .align 4 .b32 cell;

	mov.u32 	%r1, %tid.x;
	shr.u32 	%r2, %r1, 5;
	and.b32 	%r3, %r1, 31;
	setp.eq.u32 	%p1, %r3, 0;
	mov.u32 	%r5, 1;
	mov.u32 	%r6, 5;
	mov.u32 	%r7, 2;
"""

# Warp 0 stores x = 1, then u = 5; warp 1 stores v = 5, then y = 2; nothing orders the warps. A search that trusted
# the constraints x -> u, u -> v, v -> y around the obligation (u, v) would miss (x, v), (u, y) and (x, y).
BROKEN_CHAIN = """	setp.eq.u32 	%p2, %r2, 0;
	@%p2 bra 	$WARP0;
	@%p1 st.shared.u32 	[cell], %r6;
	@%p1 st.shared.u32 	[cell], %r7;
	bra.uni 	$DONE;
$WARP0:
	@%p1 st.shared.u32 	[cell], %r5;
	@%p1 st.shared.u32 	[cell], %r6;
$DONE:
	ret;
}}
"""

# Warps 0 and 1 store A = B = 1, warps 2 and 3 store C = D = 2. Barrier 2 orders B before C and D; barrier 1, which
# warp 2 reaches after barrier 2, orders A before C only. Unordered: (A, B), (C, D) and (A, D), which joins the two
# groups that the ordered constraint B -> C links.
LINKED_GROUPS = """	setp.eq.u32 	%p2, %r2, 0;
	@%p2 bra 	$WARP0;
	setp.eq.u32 	%p3, %r2, 1;
	@%p3 bra 	$WARP1;
	setp.eq.u32 	%p4, %r2, 2;
	@%p4 bra 	$WARP2;
	bra.uni 	$WARP3;
$WARP0:
	@%p1 st.shared.u32 	[cell], %r5;
	bar.arrive 	1, {count};
	bra.uni 	$DONE;
$WARP1:
	@%p1 st.shared.u32 	[cell], %r5;
	bar.arrive 	2, 96;
	bra.uni 	$DONE;
$WARP2:
	bar.sync 	2, 96;
	bar.sync 	1, {count};
	@%p1 st.shared.u32 	[cell], %r7;
	bra.uni 	$DONE;
$WARP3:
	bar.sync 	2, 96;{sync}
	@%p1 st.shared.u32 	[cell], %r7;
$DONE:
	ret;
}}
"""


def _run(directory, entry, threads, body):
    ptx = Path(directory) / f"{entry}.ptx"
    config = Path(directory) / f"{entry}.json"
    ptx.write_text(PROLOGUE.format(entry=entry, threads=threads) + body)
    config.write_text(json.dumps({
        "schema": "spverify.kernel-input/v1",
        "entry": entry,
        "launch": {"grid_dim": [1, 1, 1], "block_dim": [threads, 1, 1]},
    }))
    return verify(ptx, input=config, timeout_seconds=120)


@unittest.skipUnless(native_executable().is_file(), "native executable not built")
class SameValueObligationTests(unittest.TestCase):
    def assert_obligations(self, result, groups, writes, site_pairs, proven=False):
        if proven:
            self.assertEqual(result["result"], "race_free", result)
            self.assertEqual(result["checks"]["equivalence"], "not_requested")
        else:
            self.assertEqual(result["result"], "potential_race", result)
            self.assertEqual(result["diagnostics"][0]["code"], "same_value_writes_unproven")
        found = result["internal"]["same_value_writes"]
        self.assertEqual(found["status"], "proven" if proven else "unproven")
        self.assertEqual((found["groups"], found["writes"], found["site_pairs"]), (groups, writes, site_pairs))

    def test_pairs_around_a_broken_chain(self):
        with tempfile.TemporaryDirectory() as scratch:
            result = _run(scratch, "broken_chain", 64, BROKEN_CHAIN.format())
        # One group of the four stores: (u, v), (x, v), (u, y), (x, y); x = 1 and v = 5 differ.
        self.assert_obligations(result, groups=1, writes=4, site_pairs=4)

    def test_pair_between_linked_groups(self):
        with tempfile.TemporaryDirectory() as scratch:
            result = _run(scratch, "linked_groups", 128, LINKED_GROUPS.format(count=64, sync=""))
        # A = 1 and D = 2 differ.
        self.assert_obligations(result, groups=1, writes=4, site_pairs=3)

    def test_ordered_neighbours_keep_two_groups(self):
        # Barrier 1 also orders A before D: only (A, B) and (C, D) are unordered.
        with tempfile.TemporaryDirectory() as scratch:
            result = _run(scratch, "linked_groups_ordered", 128,
                          LINKED_GROUPS.format(count=96, sync="\n\tbar.sync \t1, 96;"))
        # A = B = 1 and C = D = 2: every unordered pair stores equal values.
        self.assert_obligations(result, groups=2, writes=4, site_pairs=2, proven=True)


if __name__ == "__main__":
    unittest.main()
